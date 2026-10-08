"""CPU stand-ins for the six tilelang kernels in DeepSeek's reference ``ref/kernel.py``.

The reference ``ref/model.py`` imports these by name from a module called ``kernel``;
``oracle.py`` installs this module under that name, so ``ref/model.py`` runs unmodified.

Each function reproduces the semantics of the tilelang kernel it replaces, read from
``ref/kernel.py`` at the pinned revision (see PROVENANCE.md). Two modes:

- ``faithful`` (default): activation and KV quantization happen exactly where the
  reference does them, with the same scale rules — this is what the reference computes,
  and the model was trained with these fake-quant points.
- ``exact``: every activation/KV quantization is skipped. Weights are still dequantized
  from their stored FP8/FP4 values, because those values *are* the model. This is the
  maximum-precision ground truth our BF16 port is compared against.

Not bit-exact with the GPU kernels: GEMMs accumulate in fp32 here but in a different
order, and sparse attention normalizes once instead of block by block. The fp4 cast is
round-to-nearest-even, which is what CUDA's ``cvt.rn`` does [unverified for tilelang's
lowering].
"""

import os

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MIDS = (_FP4_GRID[1:] + _FP4_GRID[:-1]) / 2
# low nibble is the first element, as ref/convert.py unpacks it
FP4_TABLE = torch.cat([_FP4_GRID, -_FP4_GRID])

_MODE = os.environ.get("DSV41_ORACLE_MODE", "faithful")


def set_mode(mode: str) -> None:
    global _MODE
    if mode not in ("faithful", "exact"):
        raise ValueError(f"mode must be 'faithful' or 'exact', got {mode!r}")
    _MODE = mode


def get_mode() -> str:
    return _MODE


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive normal fp32 x — the kernel's fast_round_scale exponent."""
    m, e = torch.frexp(x.float())
    # x = m * 2**e with m in [0.5, 1): an exact power of two has m == 0.5 and needs e - 1
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), e)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values already clamped to [-6, 6] onto the e2m1 grid, ties to even."""
    a = x.abs()
    mids = _FP4_MIDS.to(a.device)
    idx = torch.searchsorted(mids, a.contiguous(), right=False)
    tie = (idx < mids.numel()) & (a == mids[idx.clamp(max=mids.numel() - 1)])
    # grid entries alternate even/odd mantissa; a tie whose lower neighbour is odd goes up
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.copysign(_FP4_GRID.to(a.device)[idx], x)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[..., K//2] float4_e2m1fn_x2 or int8/uint8 -> [..., K] fp32."""
    b = packed.view(torch.uint8).long()
    table = FP4_TABLE.to(b.device)
    return torch.stack([table[b & 0x0F], table[(b >> 4) & 0x0F]], dim=-1).flatten(-2)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.float().unflatten(-1, (-1, block))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 activation quantization (fake-quant when inplace).

    Exact mode: inplace is a no-op; otherwise returns (x, None) so the GEMMs use x as is.
    """
    if _MODE == "exact":
        return x if inplace else (x, None)
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    s = pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    y = (xb / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return y.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise FP4 quantization: E8M0 power-of-two scales (indexer) or E4M3 scales
    (compressed KV). The reference only ever calls it in place."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    if _MODE == "exact":
        if inplace:
            return x
        raise NotImplementedError("non-inplace fp4_act_quant is never called by ref/model.py")
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        # training's compressed KV keeps even an all-zero group's scale nonzero
        s = (amax.clamp_min(FP4_MAX * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = pow2_ceil(amax.clamp_min(FP4_MAX * 2**-126) / FP4_MAX)
    y = round_e2m1((xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if not inplace:
        raise NotImplementedError("non-inplace fp4_act_quant is never called by ref/model.py")
    x.copy_((y * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
    return x


def _dequant_act(a, a_s, block):
    if a_s is None:  # exact mode: activation was never quantized
        return a.float()
    return (_blocks(a, block) * a_s.float().unsqueeze(-1)).flatten(-2)


def dequant_fp8_weight(b: torch.Tensor, b_s: torch.Tensor, block_size: int) -> torch.Tensor:
    """[N, K] fp8 with [ceil(N/bs), ceil(K/bs)] scales -> [N, K] fp32."""
    n, k = b.shape
    s = b_s.float().repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    return b.float() * s


def dequant_fp4_weight(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """[N, K//2] packed fp4 with [N, K//32] e8m0 scales -> [N, K] fp32."""
    w = unpack_fp4(b)
    return (_blocks(w, 32) * b_s.float().unsqueeze(-1)).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M,N] = A[M,K] @ B[N,K]^T with block-scaled FP8 operands, fp32 accumulation."""
    c = _dequant_act(a, a_s, block_size) @ dequant_fp8_weight(b, b_s, block_size).T
    return c.to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T, B packed two values per byte along K."""
    c = _dequant_act(a, a_s, act_block_size) @ dequant_fp4_weight(b, b_s).T
    return c.to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """o[b,m,h,:] = softmax over the gathered rows kv[b, topk_idxs[b,m,:]], plus a per-head
    sink logit in the denominator only. Index -1 is an empty slot. A row with no valid slot
    yields zeros, as the kernel's finite -1e30 running max guarantees.

    kv is both key and value. Faithful mode feeds the kernel's bf16 inputs and casts the
    output to bf16; exact mode keeps fp32 throughout.
    """
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = torch.gather(
        kv.float().unsqueeze(1).expand(b, m, kv.size(1), d),
        2,
        idx.clamp_min(0).unsqueeze(-1).expand(b, m, idx.size(-1), d),
    )  # [b, m, topk, d]
    qf = q.float()
    if _MODE == "faithful":
        qf, gathered = qf.bfloat16().float(), gathered.bfloat16().float()
    scores = torch.einsum("bmhd,bmtd->bmht", qf, gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    smax = scores.amax(-1).clamp_min(-1e30)  # [b, m, h]
    p = torch.exp(scores - smax.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(attn_sink.float() - smax)
    if _MODE == "faithful":
        p = p.bfloat16().float()  # the kernel feeds bf16 probabilities to the PV GEMM
    o = torch.einsum("bmht,bmtd->bmhd", p, gathered) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Split the mHC projection into pre / post / comb and Sinkhorn-normalize comb.

    Exactly ``sinkhorn_iters`` passes, ending on a column division: the loop does not
    converge, so iterating further would give a different matrix, not a better one.
    """
    hc = hc_mult
    mixes, hc_scale, hc_base = mixes.float(), hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
