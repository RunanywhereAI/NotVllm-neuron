"""Build small reference models with deterministic, realistically scaled weights.

DeepSeek's own self-test runs on torch.empty weights, which checks shapes but not numerics.
Here every parameter gets a seeded value: FP8/FP4 weights with power-of-two block scales
near 1/sqrt(fan_in), unit norms, a near-identity mHC prior (comb base 0 on the diagonal,
-8 off it, as in the released checkpoint), and zero-mean routing biases.
"""

import math

import torch

from personal_reference.deepseek_v41 import oracle

# a config that exercises every attention path at small size: window-only, ratio-2 and
# ratio-1 compressed layers, consumers reading a source's cache, and the two-level indexer
SMALL = dict(
    max_batch_size=2,
    max_seq_len=512,
    vocab_size=512,
    dim=256,
    moe_inter_dim=128,
    n_layers=6,
    n_mtp_layers=0,
    n_heads=8,
    n_routed_experts=8,
    n_activated_experts=2,
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=64,
    head_dim=64,
    rope_head_dim=16,
    o_groups=4,
    o_lora_rank=32,
    window_size=16,
    compress_ratios=(0, 2, 2, 1, 1, 1),
    kv_source_layers=(1, 3),
    index_source_layers=(1, 3, 5),
    index_n_heads=32,  # the released value; with 4, relu leaves exact-zero score ties that topk breaks arbitrarily
    index_head_dim=32,
    index_topk=8,
    candidate_source_layer=3,
    candidate_topk_blocks=3,
    candidate_block_size=4,
)


def _pow2(x: float) -> float:
    return 2.0 ** round(math.log2(x))


@torch.no_grad()
def init_weights(model: torch.nn.Module, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    params = dict(model.named_parameters())
    for name, p in params.items():
        if p.dtype == torch.float8_e8m0fnu:
            continue  # set together with its weight below
        if p.dtype == torch.float8_e4m3fn:
            fan_in = p.shape[-1]
            p.copy_(torch.randn(p.shape, generator=g).to(torch.float8_e4m3fn))
            scale = params.get(name[: -len("weight")] + "scale")
            if scale is not None:
                scale.fill_(_pow2(1 / math.sqrt(fan_in)))
        elif p.dtype == torch.float4_e2m1fn_x2:
            fan_in = p.shape[-1] * 2
            p.view(torch.uint8).copy_(torch.randint(0, 256, p.shape, generator=g, dtype=torch.uint8))
            params[name[: -len("weight")] + "scale"].fill_(_pow2(0.4 / math.sqrt(fan_in)))
        elif name.endswith("_base") and ".hc_" in name:
            hc = int(math.isqrt(p.numel() + 1) - 1)  # numel = (2 + hc) * hc
            p.zero_()
            p[2 * hc :] = (torch.eye(hc) * 8 - 8).flatten()
        elif name.endswith("_scale") and ".hc_" in name:
            p.fill_(0.1)
        elif name.endswith("_fn") and ".hc_" in name:
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
        elif "norm" in name:
            p.fill_(1.0)
        elif name.endswith("attn_sink"):
            p.copy_(torch.randn(p.shape, generator=g))
        elif name.endswith(".bias"):
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        else:
            fan_in = p.shape[-1] if p.ndim > 1 else 1
            p.copy_((torch.randn(p.shape, generator=g) / math.sqrt(fan_in)).to(p.dtype))


def build(mode: str = "exact", seed: int = 0, fix_reference_bug: bool = True, **overrides):
    """A fresh reference model (own caches) with seeded weights, on CPU."""
    m = oracle.load_reference(mode)
    args = m.ModelArgs(**{**SMALL, **overrides})
    model = oracle.build(m, args, init=lambda mod: init_weights(mod, seed), fix_reference_bug=fix_reference_bug)
    return model, args, m
