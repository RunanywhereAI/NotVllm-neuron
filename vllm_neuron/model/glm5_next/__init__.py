# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (``model_type: glm5_next``) plugin modules.

Deliberately empty of imports. The NKI kernels here must be importable on a host
that has the Neuron toolchain but not vLLM, so nothing in this package may be
pulled in as a side effect of importing a kernel. See
``personal_reference/glm5_next/sim/simulate_kernels.py``, which loads them by path.
"""
