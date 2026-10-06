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


def _layout(block_size=96, dtype=torch.bfloat16, **over):
    cfg = _Cfg()
    for k, v in over.items():
        setattr(cfg, k, v)
    return CL.LatentPageLayout.from_config(cfg, block_size, dtype)


# ------------------------------------------------------------------ the real numbers

def test_region_sizes_at_the_real_config():
    """Hand-computed from the published config, so a refactor that changes any of these
    has to change this test deliberately."""
    lay = _layout(block_size=96)
    assert lay.latent_bytes_per_token == 512 * 2          # kv_lora_rank, bf16
    assert lay.indexer_bytes_per_pool == 132              # 128 + 4 bytes inline scale
    assert lay.pools_per_page == 24                       # 96 // 4
    assert lay.latent_bytes == 96 * 1024                  # 98304
    assert lay.indexer_bytes == 24 * 132                  # 3168
    assert lay.tail_bytes == 4 * 2 * 128 * 2              # 2048
    assert lay.total_bytes == 98304 + 3168 + 2048         # 103520
    print(f"\n  {lay.describe()}")


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
        lay = _layout(block_size=b)
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


def test_odd_index_head_dim_is_rejected():
    with pytest.raises(ValueError, match="multiple of quant_block_size"):
        _layout(index_head_dim=100)


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
