# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash checkpoint -> parameters under DeepSeek's reference names.

The HF checkpoint already uses the reference's names (no ``model.`` prefix, ``attn`` /
``ffn`` / ``.scale``), so what ``ref/convert.py`` produces at ``--model-parallel 1`` is
the checkpoint's own names minus what this module drops and the scales it consumes. The
model mirrors the reference module tree one to one and loads these names unchanged.

What happens to each of the 96,085 stored tensors is decided by ``build_plan``, which
assigns every name exactly one action and raises on anything it does not recognise:

=================  ==============================================  =================
action             what                                            emitted as
=================  ==============================================  =================
``dequant_fp8``    F8_E4M3 weight, one e8m0 scale per 32x32 block  bf16, no ``.scale``
``dequant_mxfp4``  routed experts: int8 holding two e2m1 values,   bf16 ``[N, 2*K']``,
                   low nibble first, one e8m0 scale per 32 along K  no ``.scale``
``scale``          the e8m0 scale of a dequantized weight          (consumed)
``to_fp32``        bf16 in the checkpoint, fp32 in the reference:  fp32
                   ``head.weight`` and the compressor's ``wkv`` /
                   ``wgate`` on layers with ``compress_ratio > 1``
``keep``           everything else, including the Engram tables    as stored
                   (F8_E4M3 rows plus their e8m0 ``.scale``), the
                   fp32 ``attn_sink`` / ``gate.bias`` / ``hc_*``
``drop``           ``mtp.*`` (MTP and DSpark draft layers),        (not emitted)
                   ``vision.*``, ``aligner.*``, ``image_*``,
                   ``ffn.gate.bias_vl``
=================  ==============================================  =================

Both dequantizations are exact: an e4m3 value (3 mantissa bits) or an e2m1 value (1 bit)
times a power of two is representable in bf16 (7 bits) unless it leaves bf16's exponent
range, which ``_check_finite`` would catch as a non-finite result rather than pass on.
Exactness against the reference's own fp32 dequantization is a test, not an assumption.

Loading is streaming: ``Checkpoint`` opens the shards lazily and materializes one emitted
tensor (plus its scale) at a time. Two tensors are large enough to matter on their own:
each Engram table is ~98 GB of FP8 and is loaded whole by ``Checkpoint.load``; a sharded
loader should read its rows through ``Checkpoint.raw_slice`` instead.

TP / EP slicing lives in ``parallel_rule`` / ``shard`` / ``local_experts``. They mirror
the reference's ``ParallelEmbedding`` / ``ColumnParallelLinear`` / ``RowParallelLinear``
/ expert ranges and ``ref/convert.py``'s ``mapping``, and are **not wired into any
loader yet**. Note the reference's own ceiling: ``Attention.n_local_groups = o_groups //
world_size`` is 0 above TP 8, so ``wo_a`` cannot be sharded the reference's way past
``o_groups`` (8) ranks; ``shard`` refuses rather than emit a slice the reference could
not use. The indexer's ``weights_proj`` stops at ``index_n_heads`` (32).
"""

from __future__ import annotations

import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import torch

from .config import DeepseekV41TextArgs

INDEX_FILE = "model.safetensors.index.json"
FP8_BLOCK = (32, 32)   # quantization_config.weight_block_size
MX_BLOCK = 32          # e2m1 values per e8m0 scale, along K

KEEP = "keep"
TO_FP32 = "to_fp32"
DEQUANT_FP8 = "dequant_fp8"
DEQUANT_MXFP4 = "dequant_mxfp4"
SCALE = "scale"
DROP = "drop"
_EMITTING = (KEEP, TO_FP32, DEQUANT_FP8, DEQUANT_MXFP4)

# safetensors dtype strings -> torch
_ST_DTYPES = {
    "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8,
}

# e2m1 code -> value; bit 3 is the sign
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


# ------------------------------------------------------------------------ dequant
def _check_finite(name: str, t: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(t).all():
        raise ValueError(f"{name}: non-finite values after dequantization")
    return t


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor,
                      block: tuple[int, int] = FP8_BLOCK, name: str = "") -> torch.Tensor:
    """``[N, K]`` float8_e4m3fn with ``[ceil(N/bn), ceil(K/bk)]`` e8m0 scales -> bf16."""
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise TypeError(f"{name}: expected a 2-D float8_e4m3fn weight, got {weight.dtype} "
                        f"{tuple(weight.shape)}")
    n, k = weight.shape
    bn, bk = block
    expect = (-(-n // bn), -(-k // bk))
    if tuple(scale.shape) != expect:
        raise ValueError(f"{name}: scale {tuple(scale.shape)} does not tile weight "
                         f"{(n, k)} in {block} blocks (expected {expect})")
    s = scale.float()
    if n % bn == 0 and k % bk == 0:
        out = weight.float().view(n // bn, bn, k // bk, bk) * s[:, None, :, None]
        out = out.view(n, k)
    else:
        out = weight.float() * s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    return _check_finite(name, out).to(torch.bfloat16)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K']`` bytes holding two e2m1 codes each, low nibble first -> ``[..., 2K']`` fp32."""
    b = packed.view(torch.uint8)
    table = _E2M1.to(b.device)
    lo, hi = table[(b & 0x0F).long()], table[(b >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor, block: int = MX_BLOCK,
                  name: str = "") -> torch.Tensor:
    """``[N, K/2]`` int8 (or float4_e2m1fn_x2) with ``[N, K/block]`` e8m0 -> ``[N, K]`` bf16."""
    if weight.dtype not in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2) or weight.ndim != 2:
        raise TypeError(f"{name}: expected 2-D packed e2m1, got {weight.dtype} "
                        f"{tuple(weight.shape)}")
    n, k = weight.shape[0], 2 * weight.shape[1]
    if k % block or tuple(scale.shape) != (n, k // block):
        raise ValueError(f"{name}: scale {tuple(scale.shape)} does not match packed weight "
                         f"{tuple(weight.shape)} at {block} values per scale")
    out = unpack_e2m1(weight).view(n, k // block, block) * scale.float()[..., None]
    return _check_finite(name, out.view(n, k)).to(torch.bfloat16)


# --------------------------------------------------------------------------- plan
_L = r"layers\.(?P<layer>\d+)\."

# (pattern, stored dtype, action, structural requirement). COMPRESSOR resolves to
# to_fp32 / keep per layer from compress_ratios, as the reference's Compressor does.
_COMPRESSOR = "compressor"
_RULES: list[tuple[re.Pattern, str, str, str | None]] = [(re.compile(p), d, a, r) for p, d, a, r in [
    (r"embed\.weight", "BF16", KEEP, None),
    (r"head\.weight", "BF16", TO_FP32, None),
    (r"norm\.weight", "BF16", KEEP, None),
    (_L + r"(attn_norm|ffn_norm)\.weight", "BF16", KEEP, None),
    (_L + r"hc_(attn|ffn)_(fn|base|scale)", "F32", KEEP, None),
    (_L + r"attn\.attn_sink", "F32", KEEP, None),
    (_L + r"attn\.(q_norm|kv_norm)\.weight", "BF16", KEEP, None),
    (_L + r"attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight", "F8_E4M3", DEQUANT_FP8, None),
    (_L + r"attn\.compressor\.norm\.weight", "BF16", KEEP, "kv_source"),
    (_L + r"attn\.compressor\.wkv\.weight", "BF16", _COMPRESSOR, "kv_source"),
    (_L + r"attn\.compressor\.wgate\.weight", "BF16", _COMPRESSOR, "pooling"),
    (_L + r"attn\.indexer\.wq_b\.weight", "F8_E4M3", DEQUANT_FP8, "index_source"),
    (_L + r"attn\.indexer\.weights_proj\.weight", "BF16", KEEP, "index_source"),
    (_L + r"attn\.indexer\.(wk|k_norm)\.weight", "BF16", KEEP, "index_key_owner"),
    (_L + r"engram\.embed\.weight", "F8_E4M3", KEEP, "engram"),
    (_L + r"engram\.embed\.scale", "F8_E8M0", KEEP, "engram"),
    (_L + r"engram\.(q_weight|k_weight)", "BF16", KEEP, "engram"),
    (_L + r"engram\.wkv\.weight", "F8_E4M3", DEQUANT_FP8, "engram"),
    (_L + r"ffn\.gate\.weight", "BF16", KEEP, None),
    (_L + r"ffn\.gate\.bias", "F32", KEEP, None),
    (_L + r"ffn\.shared_experts\.(w1|w2|w3)\.weight", "F8_E4M3", DEQUANT_FP8, None),
    (_L + r"ffn\.experts\.(?P<expert>\d+)\.(w1|w2|w3)\.weight", "I8", DEQUANT_MXFP4, None),
]]

_DROPS: list[tuple[re.Pattern, str]] = [(re.compile(p), why) for p, why in [
    (r"mtp\..+", "MTP / DSpark draft layers: speculative decoding is out of scope"),
    (r"vision\..+", "vision tower: text-only scope"),
    (r"aligner\..+", "vision-to-text aligner: text-only scope"),
    (r"image_(start|end|newline)", "image span delimiters: text-only scope"),
    (_L + r"ffn\.gate\.bias_vl", "routing bias for image-span tokens: text-only scope"),
]]

_STRUCTURE = {
    # which layers may carry a component, from the config (reference Attention / Block)
    "kv_source": lambda a, i: i in a.kv_source_layers,
    "pooling": lambda a, i: i in a.kv_source_layers and a.compress_ratios[i] > 1,
    "index_source": lambda a, i: i in a.index_source_layers,
    "index_key_owner": lambda a, i: i in a.index_source_layers and i in a.kv_source_layers,
    "engram": lambda a, i: i in a.engram_layer_ids,
}


@dataclass(frozen=True)
class Entry:
    """What happens to one stored tensor."""

    source: str                 # checkpoint name
    action: str                 # one of the module-level action constants
    stored_dtype: str           # safetensors dtype string the rule expects
    target: str | None = None   # emitted (reference) name; None for scale / drop
    scale: str | None = None    # checkpoint name of the scale a dequant consumes
    reason: str = ""            # for drops

    @property
    def target_dtype(self) -> torch.dtype | None:
        if self.action in (DEQUANT_FP8, DEQUANT_MXFP4):
            return torch.bfloat16
        if self.action == TO_FP32:
            return torch.float32
        if self.action == KEEP:
            return _ST_DTYPES[self.stored_dtype]
        return None

    def target_shape(self, stored_shape: Iterable[int]) -> tuple[int, ...]:
        shape = tuple(stored_shape)
        if self.action == DEQUANT_MXFP4:
            return (shape[0], 2 * shape[1])
        return shape


def classify(name: str, args: DeepseekV41TextArgs) -> Entry:
    """One stored name -> its ``Entry``; ``Entry.scale`` is filled in by ``build_plan``.

    ``.scale`` names classify as ``scale`` here and are checked for a consuming weight by
    ``build_plan``; the Engram table's ``.scale`` is kept, by its own rule."""
    for pattern, why in _DROPS:
        if pattern.fullmatch(name):
            return Entry(name, DROP, "", reason=why)
    for pattern, dtype, action, need in _RULES:
        match = pattern.fullmatch(name)
        if not match:
            continue
        groups = match.groupdict()
        if "layer" in groups:
            layer = int(groups["layer"])
            if layer >= args.n_layers:
                raise ValueError(f"{name}: layer {layer} >= n_layers {args.n_layers}")
            if need and not _STRUCTURE[need](args, layer):
                raise ValueError(f"{name}: config says layer {layer} has no {need} component")
            if action == _COMPRESSOR:
                action = TO_FP32 if args.compress_ratios[layer] > 1 else KEEP
        if groups.get("expert") is not None and int(groups["expert"]) >= args.n_routed_experts:
            raise ValueError(f"{name}: expert index >= n_routed_experts {args.n_routed_experts}")
        return Entry(name, action, dtype, target=name)
    if name.endswith(".scale"):
        return Entry(name, SCALE, "F8_E8M0")
    raise ValueError(f"unrecognised DeepSeek-V4.1 tensor {name!r}")


class WeightPlan:
    """Every stored name, assigned exactly one ``Entry``."""

    def __init__(self, entries: dict[str, Entry]):
        self.entries = entries

    def emitted(self) -> list[Entry]:
        return [e for e in self.entries.values() if e.action in _EMITTING]

    def targets(self) -> set[str]:
        return {e.target for e in self.emitted()}

    def counts(self) -> Counter:
        return Counter(e.action for e in self.entries.values())

    def drop_reasons(self) -> Counter:
        return Counter(e.reason for e in self.entries.values() if e.action == DROP)


def build_plan(names: Iterable[str], args: DeepseekV41TextArgs,
               dtypes: Mapping[str, str] | None = None) -> WeightPlan:
    """Classify every stored name; pair each dequantized weight with its scale.

    Raises on an unknown name, a dequantized weight without its scale, a scale no weight
    consumes, a duplicate, and -- when ``dtypes`` (stored safetensors dtype strings, from
    ``read_header``) is given -- any tensor whose stored dtype is not what its rule says.
    """
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor names in the checkpoint index")
    if args.weight_block_size not in (None, FP8_BLOCK):
        raise NotImplementedError(f"weight_block_size {args.weight_block_size}, expected {FP8_BLOCK}")
    if args.expert_dtype not in (None, "fp4"):
        raise NotImplementedError(f"expert_dtype {args.expert_dtype!r}, expected 'fp4'")
    entries = {n: classify(n, args) for n in names}
    consumed: dict[str, str] = {}
    for n, e in list(entries.items()):
        if e.action not in (DEQUANT_FP8, DEQUANT_MXFP4):
            continue
        scale = n.removesuffix(".weight") + ".scale"
        if scale not in entries:
            raise ValueError(f"{n} is quantized but {scale} is missing")
        if entries[scale].action != SCALE:
            raise ValueError(f"{scale} is claimed by {n} but classified {entries[scale].action}")
        consumed[scale] = n
        entries[n] = Entry(n, e.action, e.stored_dtype, target=e.target, scale=scale)
    orphans = [n for n, e in entries.items() if e.action == SCALE and n not in consumed]
    if orphans:
        raise ValueError(f"{len(orphans)} scales with no dequantized weight, e.g. {orphans[:3]}")
    if dtypes is not None:
        bad = [(n, dtypes.get(n), e.stored_dtype) for n, e in entries.items()
               if e.action != DROP and dtypes.get(n) != e.stored_dtype]
        if bad:
            raise ValueError(f"{len(bad)} tensors stored in an unexpected dtype "
                             f"(name, stored, expected), e.g. {bad[:3]}")
    return WeightPlan(entries)


# ---------------------------------------------------------------------- checkpoint
def read_header(path: str | Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """name -> (safetensors dtype string, shape), from the file header only."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"}


class Checkpoint:
    """The HF checkpoint directory, read lazily, emitting reference-named tensors.

        with Checkpoint(hf_dir) as ckpt:
            for name, tensor in ckpt:          # one tensor resident at a time
                ...
    """

    def __init__(self, hf_dir: str | Path, args: DeepseekV41TextArgs | None = None):
        self.dir = Path(hf_dir)
        self.weight_map: dict[str, str] = json.loads((self.dir / INDEX_FILE).read_text())["weight_map"]
        if args is None:
            args = DeepseekV41TextArgs.from_hf_config(json.loads((self.dir / "config.json").read_text()))
        self.args = args
        self.plan = build_plan(self.weight_map, args)
        self._by_target = {e.target: e for e in self.plan.emitted()}
        self._handles: dict[str, object] = {}

    # -- raw access
    def _file(self, name: str):
        from safetensors import safe_open

        fname = self.weight_map[name]
        if fname not in self._handles:
            self._handles[fname] = safe_open(str(self.dir / fname), framework="pt", device="cpu")
        return self._handles[fname]

    def raw(self, name: str) -> torch.Tensor:
        """A stored tensor as stored."""
        return self._file(name).get_tensor(name)

    def raw_slice(self, name: str):
        """safetensors' lazy slice of a stored tensor (index it to read part of it)."""
        return self._file(name).get_slice(name)

    def headers(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        out: dict[str, tuple[str, tuple[int, ...]]] = {}
        for fname in sorted(set(self.weight_map.values())):
            out.update(read_header(self.dir / fname))
        return out

    # -- emitted tensors
    def load(self, target: str) -> torch.Tensor:
        """The tensor the model loads under reference name ``target``."""
        e = self._by_target[target]
        w = self.raw(e.source)
        if e.action == DEQUANT_FP8:
            return dequant_fp8_block(w, self.raw(e.scale), name=target)
        if e.action == DEQUANT_MXFP4:
            return dequant_mxfp4(w, self.raw(e.scale), name=target)
        if e.action == TO_FP32:
            return w.float()
        return w

    def targets(self, names: Iterable[str] | None = None) -> list[str]:
        """Emitted names grouped by shard file, optionally restricted to ``names``."""
        wanted = self._by_target.keys() if names is None else names
        missing = [n for n in wanted if n not in self._by_target]
        if missing:
            raise KeyError(f"not emitted by this checkpoint: {missing[:5]}")
        return sorted(wanted, key=lambda t: (self.weight_map[self._by_target[t].source], t))

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for t in self.targets():
            yield t, self.load(t)

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------- TP / EP (not wired)
REPLICATED, SPLIT, EXPERT, PADDED_ROWS = "replicated", "split", "expert", "padded_rows"

# reference name suffix -> split dim, from the reference's parallel classes; the same
# keys and dims as ref/convert.py's `mapping` (which matches on the module name)
_SPLITS: list[tuple[re.Pattern, int, str]] = [(re.compile(p), d, why) for p, d, why in [
    (r"embed\.weight", 0, "ParallelEmbedding: vocab rows"),
    (r"head\.weight", 0, "ParallelHead: vocab rows"),
    (_L + r"attn\.attn_sink", 0, "one sink per local head"),
    (_L + r"attn\.wq_b\.weight", 0, "ColumnParallelLinear: heads"),
    (_L + r"attn\.wo_a\.weight", 0, "ColumnParallelLinear: o_groups"),
    (_L + r"attn\.wo_b\.weight", 1, "RowParallelLinear: o_groups * o_lora_rank"),
    (_L + r"attn\.indexer\.wq_b\.weight", 0, "ColumnParallelLinear: index heads"),
    (_L + r"attn\.indexer\.weights_proj\.weight", 0, "ColumnParallelLinear: index heads"),
]]
_EXPERT_RE = re.compile(_L + r"ffn\.experts\.(?P<expert>\d+)\.(w1|w2|w3)\.weight")
_ENGRAM_ROWS_RE = re.compile(_L + r"engram\.embed\.(weight|scale)")


def parallel_rule(target: str) -> tuple[str, int | None]:
    """(kind, dim) for an emitted name: ``split`` along ``dim``, ``expert`` (whole experts
    by contiguous rank range), ``padded_rows`` (Engram table rows, ``ceil`` per rank, last
    rank padded) or ``replicated``."""
    for pattern, dim, _ in _SPLITS:
        if pattern.fullmatch(target):
            return SPLIT, dim
    if _EXPERT_RE.fullmatch(target):
        return EXPERT, None
    if _ENGRAM_ROWS_RE.fullmatch(target):
        return PADDED_ROWS, 0
    return REPLICATED, None


def local_experts(n_experts: int, rank: int, size: int) -> range:
    """The contiguous expert range a rank owns (reference ``MoE.experts_start_idx``)."""
    if n_experts % size:
        raise ValueError(f"{n_experts} experts do not divide over {size} ranks")
    per = n_experts // size
    return range(rank * per, (rank + 1) * per)


def _check_tp(target: str, size: int, args: DeepseekV41TextArgs) -> None:
    if ".attn.wo_a." in target and (size > args.o_groups or args.o_groups % size):
        raise NotImplementedError(
            f"{target}: the reference shards wo_a by whole o_groups ({args.o_groups}); "
            f"n_local_groups = o_groups // {size} is not a positive integer")
    if ".attn.indexer." in target and args.index_n_heads % size:
        raise NotImplementedError(f"{target}: {args.index_n_heads} index heads over {size} ranks")
    if (".attn.wq_b." in target or target.endswith("attn_sink")) and args.n_heads % size:
        raise NotImplementedError(f"{target}: {args.n_heads} heads over {size} ranks")


def shard(target: str, tensor: torch.Tensor | None, rank: int, size: int,
          args: DeepseekV41TextArgs) -> torch.Tensor | None:
    """This rank's part of emitted tensor ``target`` (``None`` for an expert it does not own).

    Split tensors must divide evenly, as the reference asserts; Engram rows are split
    ``ceil(rows / size)`` per rank with the last rank padded (0 for the table, 1.0 for its
    scale), as ``ref/convert.py`` does.
    """
    if not 0 <= rank < size:
        raise ValueError(f"rank {rank} not in [0, {size})")
    kind, dim = parallel_rule(target)
    if kind == REPLICATED or size == 1:
        return tensor
    if kind == EXPERT:
        e = int(_EXPERT_RE.fullmatch(target)["expert"])
        return tensor if e in local_experts(args.n_routed_experts, rank, size) else None
    if kind == PADDED_ROWS:
        rows = -(-tensor.size(0) // size)
        part = tensor[rank * rows:(rank + 1) * rows]
        if part.size(0) < rows:
            pad = tensor.new_full((rows - part.size(0), *tensor.shape[1:]),
                                  1 if target.endswith(".scale") else 0)
            part = torch.cat([part, pad])
        return part.contiguous()
    _check_tp(target, size, args)
    if tensor.size(dim) % size:
        raise ValueError(f"{target}: dim {dim} of size {tensor.size(dim)} does not divide by {size}")
    per = tensor.size(dim) // size
    return tensor.narrow(dim, rank * per, per).contiguous()


__all__ = [
    "Checkpoint", "WeightPlan", "Entry", "build_plan", "classify", "read_header",
    "dequant_fp8_block", "dequant_mxfp4", "unpack_e2m1",
    "parallel_rule", "shard", "local_experts",
    "KEEP", "TO_FP32", "DEQUANT_FP8", "DEQUANT_MXFP4", "SCALE", "DROP",
    "REPLICATED", "SPLIT", "EXPERT", "PADDED_ROWS", "FP8_BLOCK", "MX_BLOCK",
]
