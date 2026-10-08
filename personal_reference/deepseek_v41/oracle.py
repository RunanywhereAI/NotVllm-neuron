"""Load DeepSeek's reference ``ref/model.py`` on CPU with the torch kernel stand-ins.

    from personal_reference.deepseek_v41 import oracle
    m = oracle.load_reference("exact")       # or "faithful"
    model = oracle.build(m, m.ModelArgs(...))

``ref/`` is DeepSeek's code, unmodified. Its ``from kernel import ...`` resolves to
``kernel_torch`` because this module installs it under that name before the import.

Two corrections are applied to the built model rather than to ``ref/``:

1. **Stale index keys (a reference bug).** An index-key owner publishes its key cache to
   the shared runtime only inside ``if self.owns_k and latent is not None``. On a decode
   step where its compressor has not completed a group (half the steps at ratio 2), the
   shared pointer still holds the *last* owner's cache, so the indexer scores against
   another layer's keys. Measured on the small config: 27-50% logit error on those steps
   versus a fresh prefill; bit-identical once fixed. ``build`` makes owners publish on
   every call.
2. **Exact mode runs activations in fp32.** The reference computes indexer scores in
   bf16, whose 8-bit mantissa produces exact ties that ``topk`` breaks differently
   depending on tensor width. Ground truth should not depend on that, so exact mode
   builds under fp32 and upcasts every bf16 parameter. Stored FP8/FP4 weights stay as
   they are: their values are the model.
"""

import importlib
import sys
from pathlib import Path

import torch

from . import kernel_torch

REF_DIR = Path(__file__).resolve().parent / "ref"


def load_reference(mode: str = "faithful"):
    """Import ref/model.py with the stand-ins in the given mode; returns the module."""
    kernel_torch.set_mode(mode)
    sys.modules["kernel"] = kernel_torch
    if str(REF_DIR) not in sys.path:
        sys.path.insert(0, str(REF_DIR))
    if "model" in sys.modules and getattr(sys.modules["model"], "__file__", "").startswith(str(REF_DIR)):
        return sys.modules["model"]
    return importlib.import_module("model")


def fix_stale_index_keys(model, m) -> None:
    """Make every index-key owner publish its cache on every call (see module docstring)."""
    for layer in model.layers:
        ix = layer.attn.indexer
        if ix is None or not ix.owns_k:
            continue
        inner = ix.forward

        def forward(*args, _ix=ix, _inner=inner, **kwargs):
            m.shared_attn.index_k = _ix.k_cache
            return _inner(*args, **kwargs)

        ix.forward = forward


def build(m, args, init=None, fix_reference_bug: bool = True):
    """Construct ``m.Transformer(args)`` in the precision of the loaded mode.

    ``init`` (optional) fills weights before the fp32 upcast. Faithful mode keeps the
    reference's bf16 activations.
    """
    exact = kernel_torch.get_mode() == "exact"
    torch.set_default_dtype(torch.float32 if exact else torch.bfloat16)
    model = m.Transformer(args)
    if init is not None:
        init(model)
    if exact:
        for p in model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.float()
    if fix_reference_bug:
        fix_stale_index_keys(model, m)
    return model
