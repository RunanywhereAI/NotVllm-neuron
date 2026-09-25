# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) single-token decode kernel for GLM-5.3-Flash.

**STATUS: RUNS UNDER ``nki.simulate``; NEVER RUN ON A DEVICE.** Simulated 2026-09-25
on the x86 box (nki 0.6.0), BH in {2,4,6,8}, K=V=128, LNC 1 and 2. Results:

* matches the fp32 oracle (``reference.recurrent_kda`` + ``RMSNormGated``) to
  **0.53-0.66% relative**, which is the bf16 arithmetic floor -- see BF16 FLOOR below;
* **LNC1 and LNC2 are byte-identical**, including the odd-BH remainder path (BH=6),
  so the head sharding is sound;
* the scalar-gate (GDN) substitution is caught at **54.4x** the floor.

Simulation is not silicon: it does not exercise the real DMA engine, PSUM bank
allocation, or SBUF capacity. The capacity envelope under CAPACITY below is still
arithmetic, not a measurement.

PROVENANCE
----------
Adapted from nkilib ``experimental/gdn/gdn_tkg.py`` (gated DeltaNet), from the
nki-library checkout at ``/Users/aman/code/aws_infer/third_party/nki-library``
(2026-09-07, ``92d11f6``). The delta-rule core, the LNC sharding, the DMA access
patterns and the l2norm/beta folds are structurally unchanged.

nkilib's ``experimental/gdn/`` is **absent from the installed nkilib** on the dev box
(five experimental directories are missing: gdn, sparse_attention_indexer,
deepseekv32_mlp, scan, transformer), so this is vendored-from-checkout rather than
imported. In-tree precedent for carrying a kernel this way:
``vllm_neuron/functional/vendored_kernels/rotational_topk/``.

An earlier adaptation exists in the NxDI fork
(``contrib/models/GLM-5.3-Flash/src/nki_kernels/nki_kda_tkg.py``, branch
``work/qwen38-27b-and-glm53-flash``). It was read and its central insight is correct
and reused. Its header's claim that the per-channel gate is "THE ONLY MATHEMATICAL
DIFFERENCE" is **not** correct at the kernel level, and both of the differences it
misses are in Fold 4 — see WHAT CHANGED.

WHAT CHANGED vs gdn_tkg
-----------------------
1. **The gate is per-CHANNEL, not a per-head scalar.** GDN scales the ``[K, V]`` state
   by one ``exp(g)`` per head; KDA scales each K row by its own ``exp(g[k])``.

   This is nearly free here, and that is the adaptation's one elegant part. ``K`` is
   the partition axis, and ``nisa.tensor_scalar(S, S, mul, operand0=g_K_h)`` with a
   ``[K, 1]`` operand already applies per-partition. GDN fills that column with one
   broadcast scalar; KDA fills it with the real per-channel gate. **Same instruction,
   different operand contents** — and the ``stream_shuffle_broadcast`` for ``g``
   disappears, so the KDA form does marginally *less* work.

   Verified before relying on it: nkilib's own ``gdn_cte_torch_nki_ref`` and this
   oracle's ``recurrent_kda`` are **bit-identical** when the gate is held constant
   along the channel axis (``tests/test_kda.py``). So in the CORE RECURRENCE the gate
   really is the only difference. The prior art's claim is right about the recurrence
   and wrong about the kernel.

2. **Fold 4 output gate: ``sigmoid``, not ``silu``.** GDN applies ``silu(z)``; GLM's
   ``RMSNormGated`` applies ``sigmoid(z)``. This is not a variant of the same thing --
   they differ by a factor of ``z``, worth 194% relative / cosine 0.64. There is no
   config key: transformers hardcodes ``self.activation = "sigmoid"`` and vLLM passes
   ``activation="sigmoid"`` explicitly while using ``"silu"`` at three other call sites
   in the same file. The oracle had this wrong too, inherited from Qwen3.8-27B where
   ``output_gate_type: "swish"`` genuinely is silu.

3. **Fold 4 RMS epsilon: 1e-5, not 1e-6.** GDN hardcodes ``bias=1e-6``; GLM's
   ``o_norm`` is built with ``eps=config.rms_norm_eps``, which is **1e-5**.
   The l2norm epsilon in Fold 3 stays **1e-6** -- that one is correct as inherited
   (transformers' ``l2norm`` uses 1e-6). Two different epsilons, one kernel.

4. **The gate arrives precomputed.** GDN folds ``g = -exp(A_log)*softplus(a+dt_bias)``
   in-kernel from raw ``a``/``A_log``/``dt_bias``. KDA's gate is a different function
   -- ``lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`` -- and is
   per-channel, so it is computed model-side and passed as ``g_log [BH, K]``.
   Defensible, and it is what the prior art chose, but it gives up a fold and adds a
   ``[BH, K]`` fp32 HBM round-trip per layer per token. Perf item, not correctness.

KNOWN NUMERICAL NOTES
---------------------
* **l2norm spelling.** This kernel computes ``x * rsqrt(sum + eps)``; the oracle now
  computes ``x / sqrt(sum + eps)`` to match transformers. They differ by ~0.8 ULP
  (3e-8), which is ~4 orders of magnitude below the bf16 agreement floor below. Do not
  chase it; it is recorded so nobody mistakes it for a bug.
BF16 FLOOR (measured under simulation)
--------------------------------------
The kernel agrees with the fp32 oracle to **0.53-0.66% relative**. That floor comes
from **bf16 arithmetic throughout** -- ``S_bf`` is re-cast before each matmul,
``delta_bf`` and ``k_f`` are bf16 -- and NOT primarily from the bf16 ``out`` buffer.
Evidence: ``state_out`` is fp32 and carries a *comparable* floor (0.62% real,
0.59% long), so dropping the output cast buys almost nothing. There is no cheap
precision win available by validating on the state instead of the output.

Consequences for any equivalence test against this kernel:

* State it in **relative** terms. An absolute tolerance from the fp32 tests is
  meaningless.
* **The smallest detectable error is ~0.6% of the output.** A 5% perturbation of
  ``v`` or of the incoming state moves the result only 3-8x the floor. Gross
  structural errors are caught easily; subtle ones are not visible at this boundary.
* **Use the REAL gate regime.** The scalar-gate substitution clears the floor by
  **54.4x** there and only **4.2x** in the long-decay regime -- too thin to gate on.
  The polarity is opposite to the chunk-vs-recurrent carry test, so the threshold
  cannot be inherited from a neighbouring test.

CAPACITY
--------
SBUF is sized per head, and partition 0 carries four ``(1, BH_local*V)`` preloads that
are 66% of its per-head cost. With ``heads_per_rank = 64 // TP``:

    TP=1 -> batch <= 2 | TP=2 -> 5 | TP=4 -> 10 | TP=8 -> 20 | TP=64 -> 166

Binding only below TP=8, and TP=64 (one head per rank) is the recommended and maximum
configuration since TP must divide the 64 KDA heads. The preload is therefore kept
as-is rather than restructured. If TP ever drops to 4 or below, stream ``v_all``,
``z_all``, ``out_sb`` and ``k_f_all`` per head instead of preloading all of them.
Note upstream ``gdn_tkg`` has exactly ONE test case (``bh=24``), which sits at 15% of
partition 0 -- none of this envelope is covered upstream at any TP.

Math
----
    S1    = S_in * exp(g)                  [K, V]   <- exp(g) is [K, 1], PER CHANNEL
    kv    = k^T @ S1                       [1, V]
    delta = (v - kv) * beta                [1, V]
    S_out = S1 + outer(k, delta)           [K, V]
    out   = (q/sqrt(K))^T @ S_out          [1, V]
    out   = (out * rsqrt(mean_V(out^2) + 1e-5)) * norm_weight[V] * sigmoid(z)
"""

import math
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

# GLM-5.3-Flash o_norm is built with eps=config.rms_norm_eps. GDN hardcoded 1e-6.
RMS_NORM_EPS = 1e-5
# transformers' l2norm uses 1e-6; unchanged from GDN and correct as inherited.
L2NORM_EPS = 1e-6


@nki.jit
def kda_tkg(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    b: nl.NkiTensor,
    g_log: nl.NkiTensor,
    z: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    state_in: nl.NkiTensor,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Fused single-token KDA decode step.

    Folds l2norm(q,k), beta=sigmoid(b) and RMSNormGated(out, z) around the recurrent
    delta-rule core, then advances the per-head recurrent state by one token.

    Dimensions:
        BH: Flattened batch * num_heads (heads cycling fastest per batch)
        K: Key/state head dimension (<= 128; GLM uses exactly 128)
        V: Value head dimension (<= 128; GLM uses exactly 128)

    Args:
        q (nl.NkiTensor): [BH, K] @ HBM, RAW query (not l2-normed).
        k (nl.NkiTensor): [BH, K] @ HBM, RAW key (not l2-normed).
        v (nl.NkiTensor): [BH, V] @ HBM, value.
        b (nl.NkiTensor): [BH] @ HBM, RAW beta pre-sigmoid.
        g_log (nl.NkiTensor): [BH, K] @ HBM fp32, PER-CHANNEL log-space forget gate,
            ``lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``, computed
            model-side (``reference.py::ForgetGate``). The kernel applies exp() itself.
        z (nl.NkiTensor): [BH, V] @ HBM, RMSNormGated gate input (pre-sigmoid).
        norm_weight (nl.NkiTensor): [V] @ HBM, RMSNormGated weight (gamma).
        state_in (nl.NkiTensor): [BH, K, V] @ HBM fp32, incoming recurrent state.

    Returns:
        out (nl.NkiTensor): [BH, V] @ HBM, RMSNormGated decode output (bf16).
        state_out (nl.NkiTensor): [BH, K, V] @ HBM, updated recurrent state (fp32).

    Notes:
        - Per-head independent recurrence, so BH is sharded across the SPMD grid and
          each core writes only its own contiguous head slice.
        - Do NOT pre-apply exp(g) or the 1/sqrt(K) Q-scale upstream; both happen here.
        - ``g_log`` is per-channel. Passing a per-head scalar broadcast along K turns
          this into GDN and is the single most likely way to get KDA subtly wrong.
    """
    BH, K = q.shape
    _, V = v.shape
    kernel_assert(
        state_in.shape == (BH, K, V),
        f"state_in must be [BH, K, V], got {state_in.shape=}, {BH=}, {K=}, {V=}",
    )
    kernel_assert(K <= 128 and V <= 128, f"K and V must each be <= 128, got {K=}, {V=}")
    kernel_assert(
        g_log.shape == (BH, K),
        f"g_log must be PER-CHANNEL [BH, K], got {g_log.shape=}. A [BH] per-head gate "
        f"is the GDN form, not KDA.",
    )
    kernel_assert(z.shape == (BH, V), f"z must be [BH, V], got {z.shape=}")

    # ========== LNC Sharding ==========
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    BH_local = BH // n_prgs
    h_start = prg_id * BH_local
    if prg_id == n_prgs - 1:
        BH_local = BH - h_start  # last core takes the remainder

    out = nl.ndarray((BH, V), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    state_out = nl.ndarray((BH, K, V), dtype=nl.float32, buffer=nl.shared_hbm)
    scale = 1.0 / math.sqrt(K)

    # ========== Pre-allocated Buffers ==========
    S_all = nl.ndarray((K, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)
    S_bf_all = nl.ndarray((K, BH_local * V), dtype=nl.bfloat16, buffer=nl.sbuf)

    Q_p = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    K_p = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    v_all = nl.ndarray((1, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)

    exp_g_K = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    beta_K = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)

    out_sb = nl.ndarray((1, BH_local * V), dtype=nl.bfloat16, buffer=nl.sbuf)

    # ========== Bulk Preloads + Folded Pre-ops ==========
    nisa.dma_copy(dst=Q_p, src=q.ap(pattern=[[1, K], [K, BH_local]], offset=h_start * K))
    nisa.dma_copy(dst=K_p, src=k.ap(pattern=[[1, K], [K, BH_local]], offset=h_start * K))

    nisa.dma_copy(dst=v_all, src=v.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V])

    z_all = nl.ndarray((1, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=z_all, src=z.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V])
    nw_line = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nw_line, src=norm_weight.reshape((1, V)))

    k_f_all = nl.ndarray((1, BH_local * K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_f_all, src=k.reshape((1, BH * K))[:, h_start * K : h_start * K + BH_local * K])

    """
    CHANGE 1: the per-channel gate.

    GDN loads a per-head scalar onto partition 0 and broadcasts it up the partition
    axis with stream_shuffle_broadcast. KDA's gate already HAS a value per K channel,
    and K is the partition axis, so the transposed DMA below (same access pattern as
    q/k and as the state load) puts each channel's gate on its own partition directly.
    No memset -- every element is written. No broadcast -- there is nothing to
    broadcast. Everything downstream is then identical to GDN.
    """
    g_line = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_line, src=g_log.ap(pattern=[[1, K], [K, BH_local]], offset=h_start * K))
    nisa.activation(dst=exp_g_K, data=g_line, op=nl.exp)

    # beta is still a per-head scalar in KDA, so it keeps GDN's broadcast path.
    b_line = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(b_line, 0.0)
    nisa.dma_copy(dst=b_line[0:1, :], src=b.reshape((1, BH))[:, h_start : h_start + BH_local])
    stream_shuffle_broadcast(src=b_line, dst=b_line)
    nisa.activation(dst=beta_K, data=b_line, op=nl.sigmoid)

    # Fold 3: l2norm(q), l2norm(k) over K (partition axis), eps=1e-6.
    ones_K = nl.ndarray((K, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones_K, 1.0)
    q_sq = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    k_sq = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=q_sq, data=Q_p, op=nl.square)
    nisa.activation(dst=k_sq, data=K_p, op=nl.square)
    q_ss = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.psum)
    k_ss = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(q_ss, ones_K, q_sq)
    nisa.nc_matmul(k_ss, ones_K, k_sq)
    q_scale = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    k_scale = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=q_scale[0:1, :], data=q_ss, op=nl.rsqrt, bias=L2NORM_EPS)
    nisa.activation(dst=k_scale[0:1, :], data=k_ss, op=nl.rsqrt, bias=L2NORM_EPS)
    stream_shuffle_broadcast(src=q_scale, dst=q_scale)
    stream_shuffle_broadcast(src=k_scale, dst=k_scale)

    nisa.tensor_tensor(dst=K_p, data1=K_p, data2=k_scale, op=nl.multiply)

    Q_scaled = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=Q_scaled, data=Q_p, op0=nl.multiply, operand0=scale, op1=nl.multiply, operand1=q_scale
    )

    nisa.dma_copy(
        dst=S_all,
        src=state_in.ap(pattern=[[V, K], [K * V, BH_local], [1, V]], offset=h_start * K * V),
    )

    # ========== Per-head Recurrence + Folded RMSNormGated ==========
    for head_idx in range(BH_local):
        S = S_all[:, nl.ds(head_idx * V, V)]
        S_bf = S_bf_all[:, nl.ds(head_idx * V, V)]
        k_p = K_p[:, head_idx : head_idx + 1]
        q_p = Q_scaled[:, head_idx : head_idx + 1]
        v_f = v_all[:, nl.ds(head_idx * V, V)]
        k_f = k_f_all[:, nl.ds(head_idx * K, K)]
        g_K_h = exp_g_K[:, head_idx : head_idx + 1]
        b_K_h = beta_K[:, head_idx : head_idx + 1]

        nisa.tensor_scalar(k_f, k_f, op0=nl.multiply, operand0=k_scale[0:1, head_idx : head_idx + 1])

        # Step 1: S *= exp(g).
        # IDENTICAL INSTRUCTION TO GDN. g_K_h is [K,1] in both; in GDN every partition
        # holds the same broadcast scalar, here each holds its own channel's gate.
        nisa.tensor_scalar(S, S, op0=nl.multiply, operand0=g_K_h)

        nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)

        # Step 2: kv_mem = k^T @ S
        kv_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_p, k_p, S_bf)

        # Step 3: delta = (v - kv_mem) * beta
        diff = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff, data1=v_f, data2=kv_p, op=nl.subtract)
        delta = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(delta, diff, op0=nl.multiply, operand0=b_K_h[0:1, 0:1])

        # Step 4: S += outer(k, delta)
        delta_bf = nl.ndarray((1, V), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_bf, src=delta, engine=nisa.scalar_engine)
        outer_p = nl.ndarray((K, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_p, k_f, delta_bf)
        nisa.tensor_tensor(dst=S, data1=S, data2=outer_p, op=nl.add)

        # Step 5: out = q_scaled^T @ S
        nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
        out_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_p, q_p, S_bf)

        """
        Fold 4: RMSNormGated. out = (o * rsqrt(mean_V(o^2) + 1e-5)) * norm_weight * SIGMOID(z).

        CHANGE 2: sigmoid, not silu. GDN's `op=nl.silu` is correct for Qwen3.5 and
                  wrong here; the two differ by a factor of z (194% relative).
        CHANGE 3: bias=1e-5 (rms_norm_eps), not GDN's 1e-6.
        """
        o_sb = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=out_p, engine=nisa.scalar_engine)
        ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        sq = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(sq, op=nl.square, data=o_sb, reduce_op=nl.add, reduce_res=ssum)
        nisa.activation(dst=ssum, data=ssum, op=nl.rsqrt, scale=1.0 / V, bias=RMS_NORM_EPS)
        xf = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=xf,
            data=o_sb,
            op0=nl.multiply,
            operand0=ssum[0:1, 0:1],
            op1=nl.multiply,
            operand1=nw_line,
        )
        sig_z = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sig_z, data=z_all[:, nl.ds(head_idx * V, V)], op=nl.sigmoid)
        nisa.tensor_tensor(dst=out_sb[:, nl.ds(head_idx * V, V)], data1=xf, data2=sig_z, op=nl.multiply)

    # ========== Bulk Stores ==========
    nisa.dma_copy(dst=out.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V], src=out_sb)
    nisa.dma_copy(
        dst=state_out.ap(pattern=[[V, K], [K * V, BH_local], [1, V]], offset=h_start * K * V),
        src=S_all,
    )

    return out, state_out
