# SPDX-License-Identifier: Apache-2.0
"""``LatentPageLayout``, and a check that its arithmetic is not re-derived elsewhere.

Run: python3 -m pytest test/unit/test_glm5next_cache_layout.py -v -s
(needs only torch + pytest -- see ``_load_module`` for why it does not import the
package normally.)

The folded MLA page exists because the indexer page cannot be unified with the latent
page for any block size; ``cache_layout.py``'s docstring carries the derivation. What
this file adds is the part a convention cannot give you: PR #54 documents that a second
copy of page arithmetic is a **silent** memory-aliasing bug rather than a loud failure,
so ``test_offset_arithmetic_is_not_re_derived_anywhere_else`` fails if the offsets are
computed outside the one module that owns them.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAYOUT_PATH = _ROOT / "vllm_neuron" / "model" / "glm5_next" / "cache_layout.py"


def _load_module():
    """Load ``cache_layout`` from its path, bypassing ``vllm_neuron/__init__.py``.

    The package ``__init__`` pulls ``vllm``, which ships manylinux wheels only, so
    importing normally fails on macOS. ``cache_layout`` itself imports only torch, so
    loading it directly keeps this file runnable wherever the oracle suite runs. Same
    approach dev1 used to exercise ``qwen3_5/config.py`` standalone.
    """
    name = "glm5next_cache_layout"
    spec = importlib.util.spec_from_file_location(name, _LAYOUT_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before executing: ``@dataclass`` resolves ``cls.__module__`` through
    # ``sys.modules`` while processing the class, and fails with an opaque
    # ``'NoneType' has no attribute '__dict__'`` if the module is not there yet.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CL = _load_module()


class _Cfg:
    """The live GLM-5.3-Flash values, verified against the published config.json."""
    kv_lora_rank = 512
    index_head_dim = 128
    index_kpool = 4
    index_topk = 2048


def _layout(block_size=96, dtype=torch.bfloat16, indexer_fp8=False, **over):
    cfg = _Cfg()
    for k, v in over.items():
        setattr(cfg, k, v)
    lay = CL.LatentPageLayout.from_config(cfg, block_size, dtype)
    if indexer_fp8:
        import dataclasses
        lay = dataclasses.replace(lay, indexer_fp8=True)
    return lay


# ------------------------------------------------------------------ the real numbers

def test_region_sizes_at_the_real_config():
    """Hand-computed from the published config, so a refactor that changes any of these
    has to change this test deliberately. Default format: pool keys at the page dtype."""
    lay = _layout(block_size=96)
    assert lay.latent_bytes_per_token == 512 * 2          # kv_lora_rank, bf16
    assert lay.indexer_bytes_per_pool == 128 * 2          # index_head_dim, bf16
    assert lay.pools_per_page == 24                       # 96 // 4
    assert lay.latent_bytes == 96 * 1024                  # 98304
    assert lay.indexer_bytes == 24 * 256                  # 6144
    assert lay.tail_bytes == 4 * 2 * 128 * 2              # 2048
    assert lay.total_bytes == 98304 + 6144 + 2048         # 106496, no padding needed
    assert lay.total_elems % 512 == 0 and lay.total_elems % 128 == 0
    print(f"\n  {lay.describe()}")


def test_region_sizes_in_vllms_fp8_format():
    """vLLM's own format, kept as arithmetic: 128 fp8 bytes + 4 bytes of inline scale."""
    lay = _layout(block_size=96, indexer_fp8=True)
    assert lay.indexer_bytes_per_pool == 132
    assert lay.indexer_bytes == 24 * 132                  # 3168
    assert lay.total_bytes == 98304 + 3168 + 2048         # 103520
    with pytest.raises(NotImplementedError, match="FP8"):
        lay.latent_row(0, 0)


def test_offsets_tile_the_page_without_gap_or_overlap():
    lay = _layout()
    assert lay.latent_offset == 0
    assert lay.indexer_offset == lay.latent_bytes
    assert lay.tail_offset == lay.latent_bytes + lay.indexer_bytes
    assert lay.tail_offset + lay.tail_bytes == lay.total_bytes
    # the element offsets must describe the same partition
    assert lay.latent_elem_offset == 0
    assert lay.indexer_elem_offset == lay.latent_elems
    assert lay.tail_elem_offset == lay.latent_elems + lay.indexer_elems
    assert lay.tail_elem_offset + lay.tail_elems == lay.total_elems


def test_the_indexer_page_is_why_folding_is_necessary():
    """The claim the design rests on, as a test rather than a comment: the unification
    ratio is ``4096 / H`` with the block size cancelling, so no block size makes the
    indexer page divide the latent page, and 132 is the reason."""
    ratios = []
    for b in (32, 64, 96, 128, 256, 512):
        lay = _layout(block_size=b, indexer_fp8=True)
        ratios.append(lay.latent_bytes / lay.indexer_bytes)
    assert len({round(r, 9) for r in ratios}) == 1, (
        f"ratio should be block-size independent, got {ratios}"
    )
    ratio = ratios[0]
    print(f"  latent/indexer = {ratio:.6f} at every block size (4096/132)")
    assert abs(ratio - 4096 / 132) < 1e-9
    assert not float(ratio).is_integer(), "if this ever divides, folding is unnecessary"
    # and the counterfactual: without the 4 inline scale bytes it would divide exactly
    assert (4096 / 128).is_integer()


# ------------------------------------------------------------------- the guard rails

def test_block_size_must_be_a_multiple_of_index_kpool():
    """The tail region holds the open pool of the page containing the current token,
    which is only well defined when an incomplete pool cannot straddle a page."""
    _layout(block_size=96)          # 96 % 4 == 0, fine
    with pytest.raises(ValueError, match="multiple of index_kpool"):
        _layout(block_size=98)


def test_regions_stay_element_addressable():
    """Every region must be a whole number of page elements, because the page is handed
    to the model as one typed view -- Neuron rejects strided in-place writes on bound
    tensors, so a partial element would not be sliceable."""
    lay = _layout(dtype=torch.float32)
    for n in (lay.latent_bytes, lay.indexer_bytes, lay.tail_bytes):
        assert n % lay.element_size == 0


def test_odd_index_head_dim_is_rejected_in_fp8_format():
    """The inline scale count is ``index_head_dim // 128``; only the FP8 format has it."""
    with pytest.raises(ValueError, match="multiple of quant_block_size"):
        _layout(index_head_dim=100, indexer_fp8=True)
    _layout(index_head_dim=32)          # page-dtype pool keys have no scale block


# --------------------------------------------------------------- the capacity wall

def test_per_rank_latent_bytes_prints_the_wall():
    """§5's arithmetic, as a callable rather than a paragraph. The latent cache is
    replicated per rank because MLA decodes as MQA with one KV head and nothing shards
    a single head."""
    lay = _layout()
    gib = 1024 ** 3
    at_1m = lay.per_rank_latent_bytes(1_048_576, 11) / gib
    at_128k = lay.per_rank_latent_bytes(131_072, 11) / gib
    print(f"  latent per rank: {at_1m:.3f} GiB at 1M, {at_128k:.3f} GiB at 128K")
    # EXACTLY 11.0 GiB at 1M: 11 layers x 512 x 2 B = 11,264 B/token = 11 KiB, so a
    # 1M-token sequence is 11 GiB on the nose. My own documents said "~11.5 GiB";
    # writing the arithmetic as a test is what found that. 128K is 1.375 GiB.
    assert at_1m == 11.0
    assert at_128k == 1.375


# ------------------------------------------------ the enforcement master asked for

_OWNED_NAMES = (
    "latent_bytes", "indexer_bytes", "tail_bytes", "total_bytes",
    "latent_offset", "indexer_offset", "tail_offset",
    "latent_elems", "indexer_elems", "tail_elems", "total_elems",
    "latent_elem_offset", "indexer_elem_offset", "tail_elem_offset",
    "indexer_bytes_per_pool", "latent_bytes_per_token", "pools_per_page",
)


def test_offset_arithmetic_is_not_re_derived_anywhere_else():
    """The names above may be READ anywhere and DEFINED only in ``cache_layout.py``.

    PR #54's comment is the reason this is a test and not a code-review convention: a
    second copy of page arithmetic aliases memory instead of raising, so it will not
    announce itself. Reading ``layout.latent_bytes`` is fine; computing
    ``latent_bytes = block_size * 1024`` somewhere else is what this catches.
    """
    offenders: list[str] = []
    define = re.compile(
        r"^\s*(?:def\s+(" + "|".join(_OWNED_NAMES) + r")\b"
        r"|(" + "|".join(_OWNED_NAMES) + r")\s*(?::[^=]*)?=)"
    )
    for path in sorted((_ROOT / "vllm_neuron").rglob("*.py")):
        if path == _LAYOUT_PATH:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if define.match(line):
                offenders.append(f"{path.relative_to(_ROOT)}:{n}: {line.strip()}")
    assert not offenders, (
        "page-region arithmetic defined outside cache_layout.py:\n  "
        + "\n  ".join(offenders)
        + "\n\nRead these from LatentPageLayout instead. A second derivation is the "
          "silent aliasing bug PR #54 documents."
    )


def test_the_scale_constant_lives_only_in_cache_layout():
    """The 4 inline FP8 scale bytes are the whole reason folding is necessary, so the
    constant must not be duplicated or inlined elsewhere."""
    hits = [
        str(p.relative_to(_ROOT))
        for p in (_ROOT / "vllm_neuron").rglob("*.py")
        if p != _LAYOUT_PATH and "_FP8_SCALE_BYTES" in p.read_text()
    ]
    assert not hits, f"_FP8_SCALE_BYTES duplicated in: {hits}"


def test_the_enforcement_test_can_actually_fail(tmp_path):
    """A guard that cannot fail is not a guard. Plant a violation in a scratch tree and
    confirm the same matcher flags it."""
    define = re.compile(
        r"^\s*(?:def\s+(" + "|".join(_OWNED_NAMES) + r")\b"
        r"|(" + "|".join(_OWNED_NAMES) + r")\s*(?::[^=]*)?=)"
    )
    planted = [
        "    latent_bytes = block_size * 1024",
        "    def indexer_offset(self):",
        "tail_bytes: int = 2048",
    ]
    for line in planted:
        assert define.match(line), f"matcher missed a planted violation: {line!r}"
    for benign in [
        "    n = layout.latent_bytes",
        "    view = page[layout.indexer_elem_offset :]",
        "    # latent_bytes is derived in cache_layout.py",
    ]:
        assert not define.match(benign), f"matcher false-positived on: {benign!r}"


# ------------------------------------------------------------- addressing inside a page

@pytest.mark.parametrize("kvr,D,B,dtype", [
    (512, 128, 96, torch.bfloat16),     # the real config: no padding
    (64, 32, 32, torch.float32),        # the tiny test config
    (96, 64, 8, torch.float32),         # row widths that do not nest: page is padded
])
def test_row_addressing_round_trips_through_split(kvr, D, B, dtype):
    """Write a distinct value at every row the model can address, through the flat
    ``[-1, width]`` views the writes use, then read the page back through ``split``.

    Fails if any helper ignores its region offset, uses the wrong rows-per-page, lets
    two (page, token) pairs share a row, or if ``split`` disagrees with the writers --
    i.e. exactly the silent-aliasing bugs the module exists to prevent."""
    lay = _layout(block_size=B, dtype=dtype, kv_lora_rank=kvr, index_head_dim=D)
    assert lay.total_elems % kvr == 0 and lay.total_elems % D == 0
    pages = 3
    buf = torch.full((pages, lay.total_elems), -1.0, dtype=dtype)
    lat_rows, idx_rows = buf.view(-1, kvr), buf.view(-1, D)
    page = torch.arange(pages).repeat_interleave(B)
    tok = torch.arange(B).repeat(pages)
    lat_val = (page * 1000 + tok).to(dtype)
    lat_rows[lay.latent_row(page, tok)] = lat_val[:, None].expand(-1, kvr)
    first = tok % lay.index_kpool == 0                        # one write per pool
    pool_val = (page * 1000 + tok // lay.index_kpool + 500).to(dtype)
    idx_rows[lay.pool_row(page[first], tok[first])] = pool_val[first][:, None].expand(-1, D)
    in_ring = tok < lay.index_kpool                           # one write per ring slot
    k_row, g_row = lay.tail_rows(page[in_ring], tok[in_ring])
    idx_rows[k_row] = (page[in_ring] * 1000 + 700 + tok[in_ring]).to(dtype)[:, None].expand(-1, D)
    idx_rows[g_row] = (page[in_ring] * 1000 + 800 + tok[in_ring]).to(dtype)[:, None].expand(-1, D)

    latent, pools, tail = lay.split(buf)
    want_lat = (torch.arange(pages)[:, None] * 1000 + torch.arange(B)).to(dtype)
    assert torch.equal(latent, want_lat[..., None].expand(-1, -1, kvr))
    want_pool = (torch.arange(pages)[:, None] * 1000 + 500
                 + torch.arange(lay.pools_per_page)).to(dtype)
    assert torch.equal(pools, want_pool[..., None].expand(-1, -1, D))
    slots = torch.arange(lay.index_kpool)
    assert torch.equal(tail[:, :, 0], (torch.arange(pages)[:, None] * 1000 + 700 + slots)
                       .to(dtype)[..., None].expand(-1, -1, D))
    assert torch.equal(tail[:, :, 1], (torch.arange(pages)[:, None] * 1000 + 800 + slots)
                       .to(dtype)[..., None].expand(-1, -1, D))
    # nothing outside the three regions was touched: only padding is still -1
    written = (buf != -1).sum().item()
    assert written == pages * (lay.latent_elems + lay.indexer_elems + lay.tail_elems)


def test_addressing_test_detects_a_dropped_region_offset(monkeypatch):
    """Non-vacuity for the round-trip above: plant the classic bug -- a pool row that
    forgets the indexer region's offset and lands in the latent region -- and require
    the round trip to notice."""
    lay = _layout(block_size=32, dtype=torch.float32, kv_lora_rank=64, index_head_dim=32)
    bad = type(lay)
    original = bad.pool_row

    def pool_row_without_offset(self, page, token_in_page):
        per_page = self.total_elems // self.index_head_dim
        return page * per_page + token_in_page // self.index_kpool

    monkeypatch.setattr(bad, "pool_row", pool_row_without_offset)
    with pytest.raises(AssertionError):
        test_row_addressing_round_trips_through_split(64, 32, 32, torch.float32)
    monkeypatch.setattr(bad, "pool_row", original)
