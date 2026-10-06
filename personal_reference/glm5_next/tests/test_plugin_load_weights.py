# SPDX-License-Identifier: Apache-2.0
"""``Glm5NextForCausalLM.load_weights`` against an independent mapping.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_plugin_load_weights.py -v -s

A checkpoint mapping checked only against itself proves nothing: swap ``q_proj`` and
``k_proj`` in both the writer and the reader and every self-consistency test passes.
So the tiny checkpoint written here is first round-tripped through
``weight_converter.py`` -- a separate mapping, verified against all 76,108 tensor names
of the real FP8 checkpoint -- and only then read by the plugin, whose result must equal
the oracle's weights tensor for tensor and reproduce its logits.

``GLM53F_BF16_INDEX`` (optional): path to the real ``zai-org/GLM-5.3-Flash-BF16``
``model.safetensors.index.json`` (38,770 tensors). With it, the plugin's mapping at the
REAL config is checked against every real tensor name.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402
from glm5_next import weight_converter as WC  # noqa: E402
from glm5_next.tests import plugin_harness as H  # noqa: E402

CFG = H.import_plugin("vllm_neuron.model.glm5_next.config")
M = H.import_plugin("vllm_neuron.model.glm5_next.model")


@pytest.fixture(scope="module", autouse=True)
def _process_group():
    """``SafetensorsCheckpoint`` coordinates page-cache loading through the default
    distributed store, so it needs a (single-process) group."""
    import torch.distributed as dist

    if dist.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0,
                            world_size=1)
    yield
    dist.destroy_process_group()


def _oracle(seed=0):
    text = CFG.Glm5NextTextConfig.from_hf(H.hf_text_config())
    oracle = R.FlashTextModel(H.oracle_cfg(text, R), vocab=text.vocab_size).eval()
    H.randomize_(oracle, seed=seed)
    return text, oracle


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    text, oracle = _oracle()
    d = tmp_path_factory.mktemp("glm5next_tiny")
    H.write_checkpoint(H.hf_checkpoint_from_oracle(oracle.state_dict(), text.n_routed_experts), d)
    return text, oracle, d


def _bf16(sd):
    return {k: (v if k.endswith("e_score_correction_bias") else v.to(torch.bfloat16))
            for k, v in sd.items()}


def test_written_checkpoint_round_trips_through_the_converter(checkpoint, monkeypatch):
    """The writer is the converter's inverse, by the converter's own reading."""
    text, oracle, d = checkpoint
    monkeypatch.setattr(WC, "NUM_EXPERTS", text.n_routed_experts)
    got = WC.build_state_dict(d)
    want = _bf16(oracle.state_dict())
    assert set(got) == set(want)
    for k in want:
        assert torch.equal(got[k], want[k]), k


def test_load_weights_reproduces_the_oracle_weights_and_logits(checkpoint):
    text, oracle, d = checkpoint
    plugin = M.Glm5NextForCausalLM(CFG.Glm5NextConfig(text_config=text)).eval()
    plugin.load_weights(str(d), torch.device("cpu"))
    want = {k: v.float() for k, v in
            H.plugin_state_from_oracle(_bf16(oracle.state_dict())).items()}
    got = plugin.state_dict()
    assert set(got) == set(want)
    for k in want:
        assert torch.equal(got[k].float(), want[k]), k
    # and the loaded model computes the oracle's function (oracle at the same bf16 weights)
    oracle.load_state_dict({k: v.float() for k, v in _bf16(oracle.state_dict()).items()})
    run = H.FakeRunner(plugin, 32, num_blocks=8)
    ids = list(range(3, 23))
    with torch.no_grad():
        torch.testing.assert_close(run.prefill(0, ids, 32), oracle(torch.tensor([ids]))[0],
                                   rtol=2e-4, atol=2e-4)


def test_every_parameter_is_filled_and_a_missing_tensor_is_loud(checkpoint, tmp_path):
    """``strict=False`` is forced on the loader, so an unmapped parameter would keep
    its init value silently. Drop one tensor and require the load to refuse."""
    text, oracle, _ = checkpoint
    tensors = H.hf_checkpoint_from_oracle(oracle.state_dict(), text.n_routed_experts)
    del tensors["model.language_model.layers.3.self_attn.indexer.k_norm.bias"]
    H.write_checkpoint(tensors, tmp_path)
    plugin = M.Glm5NextForCausalLM(CFG.Glm5NextConfig(text_config=text))
    with pytest.raises(RuntimeError, match="k_norm.bias"):
        plugin.load_weights(str(tmp_path), torch.device("cpu"))


def test_set_dtype_keeps_the_router_bias_and_kda_decay_in_fp32():
    """The router bias chooses experts and ships F32; ``A_log``/``dt_bias`` feed the KDA
    decay. Everything else follows the serving dtype."""
    text = CFG.Glm5NextTextConfig.from_hf(H.hf_text_config())
    plugin = M.Glm5NextForCausalLM(CFG.Glm5NextConfig(text_config=text)).set_dtype(torch.bfloat16)
    for name, p in plugin.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        want = torch.float32 if leaf in M._KEEP_FP32 else torch.bfloat16
        assert p.dtype == want, (name, p.dtype)


def _real_bf16_config():
    path = pathlib.Path(os.environ.get("GLM53F_BF16_CONFIG", ""))
    if not path.is_file():
        cached = sorted(pathlib.Path.home().glob(
            ".cache/huggingface/hub/models--zai-org--GLM-5.3-Flash-BF16/snapshots/*/config.json"))
        if not cached:
            pytest.skip("real GLM-5.3-Flash-BF16 config.json not available")
        path = cached[0]
    return json.loads(path.read_text())


def test_the_real_config_parses_and_fixes_the_constants():
    """The released config through ``from_hf``, read as transformers would hand it over
    (a flat attribute namespace) -- every constant the model hardcodes or reads."""
    from types import SimpleNamespace

    raw = _real_bf16_config()["text_config"]
    text = CFG.Glm5NextTextConfig.from_hf(SimpleNamespace(**raw))
    assert (text.linear_num_heads, text.linear_head_dim, text.linear_conv_kernel_dim,
            text.linear_lower_bound) == (64, 128, 4, -5.0)
    assert (text.index_topk, text.index_kpool, text.index_n_heads, text.index_head_dim) == \
        (2048, 4, 32, 128)
    assert (text.hc_mult, text.hc_sinkhorn_iters, text.swiglu_limit,
            text.routed_scaling_factor) == (4, 20, 10.0, 2.5)
    assert text.sparse_mla_layers == tuple(range(3, 45, 4))
    assert text.mlp_layer_types[:4] == ("dense", "dense", "dense", "sparse")
    assert text.torch_dtype == torch.bfloat16


def test_mapping_covers_every_real_tensor_name():
    """At the REAL config (built on the meta device), every parameter's checkpoint keys
    exist in the real index, and every real text-decoder tensor is used."""
    index = os.environ.get("GLM53F_BF16_INDEX")
    if not index:
        pytest.skip("set GLM53F_BF16_INDEX to the real model.safetensors.index.json")
    from types import SimpleNamespace

    names = set(json.loads(pathlib.Path(index).read_text())["weight_map"])
    text = CFG.Glm5NextTextConfig.from_hf(SimpleNamespace(**_real_bf16_config()["text_config"]))
    with torch.device("meta"):
        plugin = M.Glm5NextForCausalLM(CFG.Glm5NextConfig(text_config=text))
    mapping = plugin.checkpoint_mappings()
    assert set(mapping) == {n for n, _ in plugin.named_parameters()}
    used = set()
    for keys in mapping.values():
        used |= set(keys if isinstance(keys, list) else [keys])
    missing = used - names
    assert not missing, sorted(missing)[:10]
    text_names = {n for n in names if not n.startswith(("model.visual", "visual."))
                  and ".layers.45." not in n}                      # MTP layer, descoped
    unused = text_names - used
    print(f"\n  real index: {len(names)} tensors, {len(used)} used by the plugin, "
          f"{len(names) - len(text_names)} vision/MTP dropped")
    assert not unused, sorted(unused)[:10]
