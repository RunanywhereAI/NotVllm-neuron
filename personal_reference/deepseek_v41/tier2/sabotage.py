# SPDX-License-Identifier: Apache-2.0
"""Make a deliberately wrong copy of a tiny checkpoint, so tier 2 can show it fails.

    python sabotage.py TINY_OUT_DIR SAB_OUT_DIR [SCALE_NAME]
    VLLM_NEURON_CPU_MODE=1 python run_vllm.py SAB_OUT_DIR 1 12 TINY_OUT_DIR    # must FAIL

This copies ``raw/`` and doubles one 32x32 e8m0 block scale, by default in layer 2's
shared expert, which every token passes through. It then serves the copy with
``make_served_dir``. ``run_vllm.py`` points vLLM at the copy and the oracle at the
original.

Measured 2026-10-08 at TP=1: all five prompts fail, with 6 to 11 of 12 tokens right
and worst |dlogprob| 1.95. A clean run's worst is 7.6e-6; the tolerance is 1e-3.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from vllm_neuron.model.deepseek_v41.config import make_served_dir

DEFAULT = "layers.2.ffn.shared_experts.w2.scale"


def main(src: Path, dst: Path, name: str = DEFAULT) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src / "raw", dst / "raw")
    shutil.copy2(src / "reference_args.json", dst / "reference_args.json")
    wm = json.loads((dst / "raw" / "model.safetensors.index.json").read_text())["weight_map"]
    f = dst / "raw" / wm[name]
    tensors = load_file(str(f))
    bits = tensors[name].view(torch.uint8).clone()
    print(f"{name}[0, 0]: 2^{int(bits[0, 0]) - 127} -> 2^{int(bits[0, 0]) - 126}")
    bits[0, 0] += 1
    tensors[name] = bits.view(torch.float8_e8m0fnu)
    save_file(tensors, str(f), metadata={"format": "pt"})
    make_served_dir(dst / "raw", dst / "served")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else DEFAULT)
