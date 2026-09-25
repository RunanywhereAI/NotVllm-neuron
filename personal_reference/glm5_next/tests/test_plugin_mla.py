# SPDX-License-Identifier: Apache-2.0
"""The plugin's sparse-MLA layer against the oracle.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_plugin_mla.py -v -s

The plugin computes attention in the **absorbed** form — ``q @ W_uk`` scores directly
against the 512-wide latent, and V-up is applied after attention — while the oracle
materialises K and V. Those are the same function, and that equivalence is the whole
basis for MLA decode being implementable on the existing kernel, so it is tested
rather than asserted.

Loaded by file path; see ``test_plugin_kda.py`` for why.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402


def _load(name, relpath):
    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MLA = _load("glm5next_mla", "vllm_neuron/model/glm5_next/mla.py")

H, D = 4, 96
QK, VD, KVR, QL = 32, 32, 64, 48
TOPK, KPOOL = 16, 4          # topk 16 puts the lossy regime at seq_len >= 20


class _Cfg:
    hidden_size = D
    rms_norm_eps = 1e-5
    num_attention_heads = H
    q_lora_rank = QL
    kv_lora_rank = KVR
    qk_nope_head_dim = QK
    qk_rope_head_dim = 0
    v_head_dim = VD
    index_topk = TOPK
    index_kpool = KPOOL
    index_n_heads = 32
    index_head_dim = 32


def _pair(seed=0):
    """A plugin layer and an oracle layer carrying the same weights."""
    torch.manual_seed(seed)
    m = MLA.Glm5NextSparseMLA(_Cfg(), layer_idx=3).eval()
    cfg = R.tiny_cfg(hidden_size=D, num_attention_heads=H, q_lora_rank=QL,
                     kv_lora_rank=KVR, qk_nope_head_dim=QK, v_head_dim=VD,
                     index_topk=TOPK, index_kpool=KPOOL, index_n_heads=32,
                     index_head_dim=32)
    o = R.SparseMLAttention(cfg).eval()
    o.load_state_dict(m.state_dict())
    return m, o


def _x(S, seed=0):
    return torch.randn(1, S, D, generator=torch.Generator().manual_seed(seed))


# -------------------------------------------------- absorbed == materialised K/V
@pytest.mark.parametrize("S", [8, 19, 20, 40])
def test_absorbed_attention_matches_the_oracle(S):
    """The equivalence MLA decode rests on. S spans the dense-exact ceiling (19)."""
    m, o = _pair()
    x = _x(S, seed=S)
    with torch.no_grad():
        got, latent = m.forward_core(x)
        # the oracle returns (out, (latent, indexer_state)) since it carries its own
        # indexer state; the plugin keeps state in the KV cache instead.
        want, (o_latent, _) = o(x)
    assert want.abs().max() > 1e-4, "oracle output ~zero; comparison would be vacuous"
    torch.testing.assert_close(latent, o_latent, rtol=0, atol=1e-5)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_absorption_is_not_trivially_equal():
    """Non-vacuity: the absorbed path must really be a different computation, not the
    materialised one under another name."""
    m, _ = _pair()
    W_uk, W_uv = m._uk_uv()
    assert W_uk.shape == (H, QK, KVR) and W_uv.shape == (H, VD, KVR)
    x = _x(12, seed=2)
    with torch.no_grad():
        q_c = m.q_a_layernorm(m.q_a_proj(x))
        q = m.q_b_proj(q_c).view(1, 12, H, QK)
        latent = m.kv_a_layernorm(m.kv_a_proj_with_mqa(x)[..., :KVR])
        # absorbed scores, against the latent directly
        q_abs = torch.einsum("bshq,hqr->bshr", q.float(), W_uk.float())
        a = torch.einsum("bshr,blr->bhsl", q_abs, latent.float())
        # materialised scores, against a lifted K
        k = torch.einsum("blr,hqr->blhq", latent.float(), W_uk.float())
        b = torch.einsum("bshq,blhq->bhsl", q.float(), k)
    assert q_abs.shape[-1] == KVR, "absorbed query must live in the LATENT width"
    assert q.shape[-1] == QK != KVR, "the two widths must differ or this proves nothing"
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------ config guards
def test_nope_is_required():
    """A non-zero rope dim makes the latent 576 wide, which fails the decode kernel's
    d_head checks. This is exactly why DeepSeek's MLA cannot use this path."""
    class Bad(_Cfg):
        qk_rope_head_dim = 64
    with pytest.raises(ValueError, match="NoPE-only"):
        MLA.MLAParams.from_config(Bad())


def test_index_n_heads_must_be_32():
    """64 is DeepSeek's value, and vLLM's source carries a stale '# 64' comment three
    lines from the code that reads the config."""
    class Bad(_Cfg):
        index_n_heads = 64
    with pytest.raises(ValueError, match="DeepSeek"):
        MLA.MLAParams.from_config(Bad())


def test_the_latent_is_one_tensor_serving_as_both_k_and_v():
    """Reading may alias — the property that makes a single latent cache sound."""
    m, _ = _pair()
    x = _x(10, seed=4)
    with torch.no_grad():
        _, latent = m.forward_core(x)
    assert latent.shape == (1, 10, KVR), "one latent per token, not separate K and V"


# ---------------------------------------------------------------- the indexer
def test_indexer_selects_everything_below_the_ceiling():
    """topk 16, kpool 4 -> exact to 19 = topk + kpool - 1, NOT topk. vLLM's own gate
    at topk is conservative, and the difference is the always-selected tail."""
    m, _ = _pair()
    x = _x(19, seed=6)
    with torch.no_grad():
        idx = m.indexer(x, m.q_a_layernorm(m.q_a_proj(x)))
    mask = MLA.indices_to_mask(idx.long(), 19)[0]
    causal = torch.ones(19, 19, dtype=torch.bool).tril()
    assert torch.equal(mask, causal), "must be exactly causal at or below the ceiling"


def test_indexer_drops_tokens_above_the_ceiling():
    m, _ = _pair()
    S = 40
    x = _x(S, seed=7)
    with torch.no_grad():
        idx = m.indexer(x, m.q_a_layernorm(m.q_a_proj(x)))
    mask = MLA.indices_to_mask(idx.long(), S)[0]
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    assert not (mask & ~causal).any(), "selected a future token"
    dropped = int((causal & ~mask).any(-1).sum())
    print(f"\n  S={S}: {dropped} rows drop tokens (ceiling is {TOPK + KPOOL - 1})")
    assert dropped == S - (TOPK + KPOOL - 1)


def test_only_complete_pools_are_scored():
    """A pool holding future tokens must never be a candidate. Perturbing future
    tokens must not move an earlier row's selection."""
    m, _ = _pair()
    S, t = 40, 24
    x = _x(S, seed=8)
    x2 = x.clone()
    x2[:, t:] += 3 * torch.randn_like(x2[:, t:])
    with torch.no_grad():
        qc = lambda z: m.q_a_layernorm(m.q_a_proj(z))
        a = MLA.indices_to_mask(m.indexer(x, qc(x)).long(), S)
        b = MLA.indices_to_mask(m.indexer(x2, qc(x2)).long(), S)
    assert torch.equal(a[:, :t], b[:, :t]), "future tokens moved an earlier selection"
    assert not torch.equal(a[:, t:], b[:, t:]), "perturbation had no effect at all"


# ------------------------------------------------- the override and capture points
def test_topk_override_changes_attention_without_skipping_the_indexer(monkeypatch):
    """The validation affordance. Injecting a selection must drive the mask, and the
    indexer must still run so its state and captures still advance."""
    m, _ = _pair()
    S = 40
    x = _x(S, seed=9)
    seen = {}
    monkeypatch.setattr(MLA, "_capture_tensor", lambda n, t: seen.__setitem__(n, t))
    with torch.no_grad():
        base, _ = m.forward_core(x)
        causal = torch.arange(S).expand(1, S, S).masked_fill(
            ~torch.ones(S, S, dtype=torch.bool).tril(), -1)
        forced, _ = m.forward_core(x, topk_indices=causal)
    assert f"{m.layer_name}.indexer.scores" in seen, "override skipped the indexer"
    rel = (forced - base).abs().max() / base.abs().max()
    print(f"  forcing the full causal set moves the output {rel:.3f} relative")
    assert rel > 1e-3, "the override did not drive the mask"


def test_override_is_unreachable_from_the_framework_path():
    """It must not be possible to leave the override on in a serving path."""
    import inspect
    sig = inspect.signature(MLA.Glm5NextSparseMLA.forward)
    assert "topk_indices" not in sig.parameters
    assert "topk_indices" in inspect.signature(MLA.Glm5NextSparseMLA.forward_core).parameters


def test_capture_points_fire_and_carry_tensors(monkeypatch):
    seen = {}
    monkeypatch.setattr(MLA, "_capture_tensor", lambda n, t: seen.__setitem__(n, t))
    m, _ = _pair()
    with torch.no_grad():
        m.forward_core(_x(24, seed=11))
    for want in ("latent", "attn_pre_oproj", "indexer.scores", "indexer.topk_indices"):
        name = f"{m.layer_name}.{want}"
        assert name in seen, f"{name} never fired; captured {sorted(seen)}"
        assert isinstance(seen[name], torch.Tensor) and seen[name].numel() > 0
    # scores are the point: exact index equality fails a correct fp8 device
    assert seen[f"{m.layer_name}.indexer.scores"].dtype == torch.float32
    print(f"\n  captured: {sorted(n.split(m.layer_name + '.')[-1] for n in seen)}")


def test_layer_imports_without_vllm():
    assert "vllm" not in sys.modules
