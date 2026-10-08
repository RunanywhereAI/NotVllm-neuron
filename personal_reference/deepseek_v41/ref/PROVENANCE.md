# Provenance

DeepSeek's reference inference code for DeepSeek-V4.1-Flash, vendored unmodified from
`huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`, `inference/`, at repo revision
`2cba9e42aa026125f3ed06c6d98c1db82f7ca027` (fetched 2026-10-08).

Do not edit these files. `kernel.py` is kept for reference only: it needs tilelang on CUDA,
so `../oracle.py` installs `../kernel_torch.py` under the module name `kernel` instead.
Corrections to the reference are applied to built models in `../oracle.py`, where each is
documented and pinned by a test.
