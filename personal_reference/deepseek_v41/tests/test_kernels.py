"""The kernel stand-ins against independent formulations, not against themselves.

Each test names what a wrong implementation would get wrong and checks that case.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from personal_reference.deepseek_v41 import kernel_torch as K  # noqa: E402
from personal_reference.deepseek_v41.ref import convert as ref_convert  # noqa: E402
from personal_reference.deepseek_v41.tests.vllm_refs import mhc_pre_torch  # noqa: E402


@pytest.fixture(autouse=True)
def faithful():
    K.set_mode("faithful")
    yield
    K.set_mode("faithful")


def test_pow2_ceil_matches_float_log2_including_exact_powers():
    # an exact power of two must NOT round up: that is the bit the kernel checks
    xs = torch.tensor([2.0**-20, 0.25, 0.3, 0.5, 1.0, 1.0000001, 3.0, 4.0, 447.9, 448.0])
    want = torch.tensor([2.0 ** math.ceil(math.log2(v)) for v in xs.tolist()])
    assert torch.equal(K.pow2_ceil(xs), want)


def test_round_e2m1_is_nearest_with_ties_to_even():
    # brute force: nearest grid point, ties to the even-mantissa neighbour
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    even = {0.0, 1.0, 2.0, 4.0}
    xs = torch.linspace(-6, 6, 4801)
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    xs = torch.cat([xs, torch.tensor(ties + [-t for t in ties])])

    def ref(v):
        a = abs(v)
        best = min(grid, key=lambda g: (abs(g - a), g not in even))
        return math.copysign(best, v)

    want = torch.tensor([ref(v) for v in xs.tolist()])
    assert torch.equal(K.round_e2m1(xs), want)
    # the ties specifically, since linspace may never land on them
    assert K.round_e2m1(torch.tensor(ties)).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]


def test_unpack_fp4_agrees_with_reference_convert():
    # ref/convert.py's lossless e2m1 -> e4m3 cast is an independent unpacking of the same bytes
    torch.manual_seed(0)
    packed = torch.randint(-128, 128, (64, 32), dtype=torch.int8)  # 64 x 64 fp4 values
    # convert.py's cast is lossless only while a 32x32 block's scales span <= 2**6 (its
    # MAX_OFFSET_BITS); wider spans underflow fp8. Keep the span inside that, as real weights do
    # [unverified for every real block].
    scale = torch.randint(124, 131, (64, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ours = K.dequant_fp4_weight(packed, scale)
    w8, s8 = ref_convert.cast_e2m1fn_to_e4m3fn(packed.clone(), scale.clone())
    theirs = K.dequant_fp8_weight(w8, s8, 32)
    assert torch.equal(ours, theirs)


def test_unpack_fp4_nibble_order_is_low_first():
    b = torch.tensor([[0x72]], dtype=torch.uint8)  # low nibble 2 -> 1.0, high nibble 7 -> 6.0
    assert K.unpack_fp4(b).tolist() == [[1.0, 6.0]]


def test_act_quant_pow2_scales_and_error_bound():
    torch.manual_seed(1)
    x = (torch.randn(8, 256) * torch.logspace(-3, 3, 8).unsqueeze(1)).bfloat16()
    y, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    sf = s.float()
    assert torch.equal(sf, K.pow2_ceil(sf))  # every scale a power of two
    deq = (y.float().unflatten(-1, (-1, 32)) * sf.unsqueeze(-1)).flatten(-2)
    # e4m3 has 3 mantissa bits: relative error <= 2**-4 for normals, absolute floor for subnormals
    err = (deq - x.float()).abs()
    bound = x.float().abs() * 2**-4 + sf.repeat_interleave(32, -1) * 2**-9
    assert (err <= bound).all()
    # the pow2 scale never overflows the fp8 range, so nothing clamped to NaN
    assert torch.isfinite(deq).all()


def test_act_quant_inplace_modifies_and_exact_mode_does_not():
    x = torch.randn(4, 64).bfloat16()
    orig = x.clone()
    K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    assert not torch.equal(x, orig)
    K.set_mode("exact")
    x2 = orig.clone()
    K.act_quant(x2, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    assert torch.equal(x2, orig)
    a, s = K.act_quant(orig, 32, "ue8m0", torch.float8_e8m0fnu)
    assert s is None and a is orig


@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.float8_e4m3fn])
def test_fp4_act_quant_lands_on_scaled_grid(scale_dtype):
    torch.manual_seed(2)
    x = torch.randn(16, 128).bfloat16() * 3
    q = x.clone()
    K.fp4_act_quant(q, 32 if scale_dtype == torch.float8_e8m0fnu else 16, True, scale_dtype)
    block = 32 if scale_dtype == torch.float8_e8m0fnu else 16
    qb = q.float().unflatten(-1, (-1, block))
    xb = x.float().unflatten(-1, (-1, block))
    # each block is (grid value) * one scale: dividing by the block's scale gives grid points
    amax = xb.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        s = K.pow2_ceil(amax / 6)
    on_grid = qb / s.unsqueeze(-1)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.isin(on_grid.abs(), grid).all()
    # and it is a real quantization: some values moved
    assert not torch.equal(q, x)


def test_sparse_attn_against_dense_softmax_with_sink_column():
    # independent form: append the sink as an extra logit whose value row is zero
    torch.manual_seed(3)
    b, m, h, d, n, topk = 2, 3, 4, 64, 10, 6
    q, kv = torch.randn(b, m, h, d), torch.randn(b, n, d)
    sink = torch.randn(h)
    idx = torch.randint(0, n, (b, m, topk), dtype=torch.int32)
    idx[0, 1, 4:] = -1  # empty slots
    K.set_mode("exact")
    got = K.sparse_attn(q, kv, sink, idx, 0.125)
    want = torch.zeros_like(got)
    for bi in range(b):
        for mi in range(m):
            sel = [int(i) for i in idx[bi, mi] if i >= 0]
            keys = kv[bi, sel]
            logits = torch.cat([q[bi, mi] @ keys.T * 0.125, sink.unsqueeze(-1)], -1)
            p = torch.softmax(logits, -1)
            want[bi, mi] = p[:, :-1] @ keys
    assert torch.allclose(got, want, atol=1e-5)
    # the sink matters: dropping it changes the output
    no_sink = K.sparse_attn(q, kv, torch.full((h,), -1e30), idx, 0.125)
    assert (no_sink - got).abs().max() > 1e-2


def test_sparse_attn_all_empty_row_is_zero_not_nan():
    q, kv = torch.randn(1, 1, 2, 8), torch.randn(1, 4, 8)
    out = K.sparse_attn(q, kv, torch.zeros(2), torch.full((1, 1, 3), -1, dtype=torch.int32), 1.0)
    assert torch.equal(out, torch.zeros_like(out))


def test_hc_mixes_plus_sinkhorn_agree_with_vllm_mhc_pre():
    # ref Block.hc_mixes = rms-normalized projection, then our hc_split_sinkhorn; vLLM fuses both
    torch.manual_seed(4)
    hc, dim, tokens = 4, 32, 5
    residual = torch.randn(tokens, hc, dim).bfloat16()
    fn = torch.randn((2 + hc) * hc, hc * dim)
    scale, base = torch.rand(3) + 0.5, torch.randn((2 + hc) * hc)
    x = residual.flatten(1).float()
    mixes = torch.nn.functional.linear(x, fn) * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    pre, post, comb = K.hc_split_sinkhorn(mixes.unsqueeze(0), scale, base, hc, 20, 1e-6)
    v_post, v_comb, v_in = mhc_pre_torch(residual, fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20)
    assert torch.allclose(post[0], v_post.squeeze(-1), atol=1e-6)
    assert torch.allclose(comb[0], v_comb, atol=1e-6)
    ours_in = (pre[0].unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    assert torch.equal(ours_in, v_in)


def test_sinkhorn_is_truncated_and_ends_on_a_column_division():
    # at the real near-identity prior (comb base 0 on the diagonal, -8 off it) the loop is still
    # short of convergence at 20: columns are normalized (last op) but rows are not
    torch.manual_seed(5)
    base = torch.zeros(24)
    base[8:] = (torch.eye(4) * 8 - 8).flatten()
    mixes, scale = torch.randn(1, 3, 24), torch.ones(3)
    _, _, c20 = K.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    _, _, c21 = K.hc_split_sinkhorn(mixes, scale, base, 4, 21, 1e-6)
    col_err = (c20.sum(-2) - 1).abs().max()
    row_err = (c20.sum(-1) - 1).abs().max()
    assert col_err < 1e-5 and row_err > 100 * col_err
    # one more iteration changes the matrix: reproduce exactly 20, never "converge harder"
    assert (c20 - c21).abs().max() > 1e-5


def test_fp8_gemm_equals_dequantized_matmul_on_representable_inputs():
    torch.manual_seed(6)
    w = torch.randn(64, 96).to(torch.float8_e4m3fn)
    ws = torch.tensor([[0.5, 2.0, 0.5], [2.0, 0.5, 2.0]]).to(torch.float8_e8m0fnu)  # [64/32, 96/32]
    a = torch.randn(5, 96).bfloat16()
    K.set_mode("exact")
    a_q, a_s = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
    got = K.fp8_gemm(a_q, a_s, w, ws, torch.float8_e8m0fnu, 32)
    blockscale = ws.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    want = (a.float() @ (w.float() * blockscale).T).to(torch.get_default_dtype())
    assert torch.equal(got, want)
