# SPDX-License-Identifier: Apache-2.0
"""``state_page_indices``, pinned against the inline version it was lifted from.

Run: python3 -m pytest test/unit/test_kv_cache_state_indices.py -v -s

The logic was lifted out of ``qwen3_5/deltanet.py``, which now delegates, so Qwen3.8-27B
depends on this being behaviourally identical. Qwen has no test suite of its own on this
branch, so the characterisation test lives here: ``_inline_original`` is the pre-lift
body verbatim, and every case below asserts the two agree.

What the pair exists for, since it reads like redundancy: a padded batch row must READ
the zero page, because its logits are not discarded and the sampler's argmax reduces
across the whole tile -- a dead row reading another cache group's bytes as float32 hands
NaN to every live row. It must WRITE somewhere else, because writing the zero page is
what would stop it being zeros.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # @dataclass needs this before exec
    spec.loader.exec_module(mod)
    return mod


KC = _load("glm5next_kv_cache", "vllm_neuron/model/kv_cache.py")


def _inline_original(metadata, num_reqs, num_pages):
    """``Qwen3_5DeltaNet.state_indices``'s body as it stood before the lift."""
    block_table = metadata["block_table_tensor"]
    indices = block_table[:num_reqs, 0].to(torch.long)
    slot_mapping = metadata["slot_mapping"].view(num_reqs, -1)[:, 0]
    sink = num_pages - 1
    zero_page = sink - 1
    live = (slot_mapping > 0) & (indices > 0) & (indices < zero_page)
    return torch.where(
        live, indices, torch.full_like(indices, zero_page)
    ), torch.where(
        live, indices, torch.full_like(indices, sink)
    )


def _meta(block_ids, slots, tokens_per_req=1):
    return {
        "block_table_tensor": torch.tensor(block_ids, dtype=torch.int32).view(-1, 1),
        "slot_mapping": torch.tensor(slots, dtype=torch.int32).view(
            len(block_ids), tokens_per_req
        ),
    }


_CASES = {
    "all live": ([3, 5, 7], [96, 160, 224]),
    "one padded by slot": ([3, 5, 7], [96, 0, 224]),
    "one padded by null block": ([3, 0, 7], [96, 160, 224]),
    "stale out-of-range id": ([3, 998, 7], [96, 160, 224]),
    "id exactly at zero page": ([3, 8, 7], [96, 160, 224]),
    "all padded": ([0, 0, 0], [0, 0, 0]),
    "negative legacy sentinel": ([3, 5, 7], [96, -1, 224]),
    "single request": ([4], [128]),
}


@pytest.mark.parametrize("label", list(_CASES))
def test_matches_the_inline_original(label):
    """Behaviour-identical to the version Qwen still depends on."""
    block_ids, slots = _CASES[label]
    num_pages = 10          # so sink=9, zero_page=8
    meta = _meta(block_ids, slots)
    got_r, got_w = KC.state_page_indices(meta, len(block_ids), num_pages)
    exp_r, exp_w = _inline_original(_meta(block_ids, slots), len(block_ids), num_pages)
    assert torch.equal(got_r, exp_r), f"{label}: read pages differ"
    assert torch.equal(got_w, exp_w), f"{label}: write pages differ"
    print(f"  {label:26} read={got_r.tolist()} write={got_w.tolist()}")


def test_padded_rows_read_zero_and_write_the_sink():
    """The property the pair exists for, asserted directly rather than via the original.

    If these ever coincide, a padded row writes the page every other padded row reads,
    and the zero page stops being zeros.
    """
    read, write = KC.state_page_indices(_meta([3, 0], [96, 0]), 2, 10)
    assert read.tolist() == [3, 8]      # zero_page = num_pages - 2
    assert write.tolist() == [3, 9]     # sink = num_pages - 1
    assert read[1] != write[1], "padded row must not write the page it reads"


def test_null_block_zero_is_treated_as_padding_not_page_zero():
    """The vLLM 0.24 sentinel change. ``>= 0`` here instead of ``> 0`` would route a
    padded row to real page 0 and corrupt whichever request owns it."""
    read, write = KC.state_page_indices(_meta([0], [0]), 1, 10)
    assert read.tolist() == [8] and write.tolist() == [9]


def test_out_of_range_id_is_bounded_not_wrapped():
    """A stale block id from an earlier step must be redirected, not indexed. On device
    an out-of-range index is an out-of-bound indirect DMA rather than a wrapped one."""
    read, write = KC.state_page_indices(_meta([12345], [96]), 1, 10)
    assert read.tolist() == [8] and write.tolist() == [9]


def test_only_the_first_token_of_each_request_is_consulted():
    """State is per sequence, not per token, so multi-token rows use column 0."""
    meta = _meta([3, 5], [96, 0, 160, 0], tokens_per_req=2)
    read, _ = KC.state_page_indices(meta, 2, 10)
    assert read.tolist() == [3, 5]


def test_the_characterisation_test_can_fail():
    """A drift detector that cannot fire proves nothing. Perturb the shared function's
    contract and confirm the comparison notices."""
    meta = _meta([3, 0], [96, 0])
    got_r, _ = KC.state_page_indices(meta, 2, 10)
    wrong_r, _ = _inline_original(_meta([3, 0], [96, 0]), 2, 11)   # wrong num_pages
    assert not torch.equal(got_r, wrong_r), (
        "comparison is insensitive to num_pages, so it would not catch a drift in it"
    )


def test_paged_block_ids_sends_every_unusable_entry_to_the_zero_page():
    """The decode gather's redirect. Live entries pass; a dead row, the null block, a
    stale id at or past the reserved pages, and a negative sentinel all read zeros."""
    num_pages = 10                       # 8 real blocks + zero page 8 + sink 9
    zero_page, sink = KC.reserved_pages(num_pages)
    assert (zero_page, sink) == (8, 9)
    bt = torch.tensor([[3, 5, 0, -1], [7, 8, 9, 12], [2, 4, 6, 1]], dtype=torch.int32)
    live = torch.tensor([True, True, False])
    got = KC.paged_block_ids(bt, live, num_pages)
    want = torch.tensor([[3, 5, 8, 8], [7, 8, 8, 8], [8, 8, 8, 8]])
    assert torch.equal(got, want)
    # and it must agree with state_page_indices about which page is the zero page
    meta = _meta([3, 4], [96, 0])
    read, write = KC.state_page_indices(meta, 2, num_pages)
    assert read[1].item() == zero_page and write[1].item() == sink
