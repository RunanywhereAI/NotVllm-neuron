# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) chunked-prefill kernel for GLM-5.3-Flash.

**STATUS: RUNS UNDER ``nki.simulate``; NEVER RUN ON A DEVICE.** Simulated 2026-09-25,
BH 1 and 2, S = 64/128/192 (so the inter-chunk state carry is exercised), D = 128:

* vs ``reference.chunk_kda``: output **0.62-0.78%**, state **0.40-0.57%** relative --
  the same bf16 arithmetic floor measured for the decode kernel (0.36-0.66%).
* Feeding a per-head SCALAR gate moves the output **24.5x** that floor, so the kernel
  genuinely consumes the channel axis rather than collapsing it.
* And with a genuinely scalar gate it still matches the reference (0.73%), so it is
  correct in both regimes rather than merely sensitive to the gate's shape.

Two transcription bugs found by simulating against the torch reference, both worth
knowing because neither is visible by reading:

1. ``nisa.tensor_tensor`` cannot take **both** operands from PSUM. Two sites needed an
   eviction to SBUF first.
2. The intra-chunk output mask keeps ``j <= i`` **inclusive**, while the ``A`` operator
   keeps ``j < i`` strictly. Using the strict mask for both dropped the diagonal term.
   The signature was sharp: **state correct, output 99.8% wrong** -- because the state
   path does not read the intra mask, and on chunk 0 the state is still zero, so row 0
   of the output was exactly zero. ``chunk_kda`` spells the difference as
   ``masked_fill(tri)`` vs ``masked_fill(stri)``; ``gdn_cte`` as ``offset=-1`` vs
   ``offset=0``.

PROVENANCE
----------
Adapted from nkilib ``experimental/gdn/gdn_cte.py`` (chunked gated delta-rule),
checkout ``92d11f6``. The chunk loop, the nilpotent solve and the state carry keep
that file's shape. The algorithm implemented here is validated in torch as
``tests/test_kda_prefill_anchor.py::chunk_kda_anchored``, which reproduces
``reference.chunk_kda`` to <1e-5 with no ``[c,c,K]`` intermediate. **This file is a
transcription of that validated algorithm**, so a disagreement with it is a
transcription bug, not a design question.

WHAT CHANGED vs gdn_cte, and it is four things, not one
-------------------------------------------------------
GDN's gate is one scalar per token (``gate: [B, S]``). KDA's is per channel
(``gate: [B, S, D]``). That single fact breaks four separate things:

1. **The intra-chunk operator does not factor.** GDN builds
   ``decay[i,j] = exp(cg[i] - cg[j])`` as a ``[CHUNK, CHUNK]`` matrix and forms
   ``A = -beta * (k @ k^T) * decay``. KDA needs

       A[i,j] = -beta[i] * sum_d k[i,d] k[j,d] exp(cg[i,d] - cg[j,d])

   with the decay **inside the d-sum**, where it does not come out. Done literally it
   needs ``[CHUNK, CHUNK, D]`` = 2 MB per chunk per head. Solved by **anchoring**: see
   ANCHORING below.

2. **The same problem at the intra-chunk output** (``qkt_decay`` in gdn_cte). It is the
   same bilinear form with ``q`` in place of ``k_beta``, so one helper covers both --
   and those are the *only* two sites that need it.

3. **``exp(cg)`` no longer commutes out of the matmul.** gdn_cte computes
   ``o_cross = (q @ state) * exp_cg`` because a per-token scalar commutes past the
   contraction. Per channel it does **not**: the correct order is
   ``(q * exp_cg) @ state``. Measured on the reordering: **124% relative error**, and
   exactly 0 when the gate is per-token, confirming it is a GDN-specific optimisation
   rather than a general identity. Same for ``k_cumdecay``.

4. **The state decay becomes a per-partition operand.** ``state * exp(g_last)`` with
   ``state`` as ``[D_k, D_v]`` and ``D_k`` on the partition axis makes ``exp(g_last)``
   a genuine ``[D, 1]`` column. This one is **free** -- the same instruction GDN uses,
   with real per-channel values instead of a broadcast scalar.

Everything else in the path is already safe: ``exp(cg)``, ``exp(cg_last - cg)`` and
``exp(cg_last)`` are all <= 1 and underflow benignly (a fully decayed state
contributes nothing, which is the right answer). **The overflow bound below applies to
exactly one expression in this kernel.**

ANCHORING, and where _SUBBLK comes from
---------------------------------------
Factor the pairwise decay through an anchor at each row-block's first token::

    exp(cg[i,d] - cg[j,d]) = exp(cg[i,d] - cg_n[d]) * exp(cg_n[d] - cg[j,d])

Pre-scale ``k`` on each side and ``A = -(Kb' @ K''^T)`` is a plain matmul again.

* Row factors (``i >= n``) are **<= 1**; underflow is benign.
* Column factors on the **diagonal block** (``j >= n``) are **>= 1** and grow with the
  distance to the anchor. **This is the binding side, and neither FLA nor nkilib says
  so.**
* Worst case ``exp(_SUBBLK * |gate_lower_bound|)``, so safety needs
  ``_SUBBLK * 5.0 < ln(float32_max) = 88.7`` -> ``_SUBBLK <= 17``, and the largest
  usable power of two is **16**.

So 16 here is **derived from ``gate_lower_bound``**, not inherited. nkilib's
``_SUBBLK = 16`` is sized for its nilpotent solve and FLA's ``BC = min(16, BT)`` is
unexplained in source; both land on 16 for reasons that are not this one. It is a
**function of a config value**: ``gate_lower_bound = -6.0`` would halve the usable
size to 8 (bound 14, largest power of two 8), which is a re-tiling rather than a
constant change. fp32 and bf16 give the same bound -- they share an exponent range.

The anchor's *position* is algebraically irrelevant (the factors telescope for any
anchor); it only sets the range each factor must span. Anchoring a whole chunk at
token 0 overflows exactly as ``_SUBBLK = 64`` does.
"""

import math
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

_CHUNK = 64
_SUBBLK = 16                      # derived above from gate_lower_bound = -5.0
_N_SUBBLK = _CHUNK // _SUBBLK
# (I - A)^-1 for a strictly-lower-triangular [CHUNK, CHUNK] A, which is nilpotent with
# A^CHUNK = 0: M = prod_i (I + A^(2^i)) over ceil(log2(CHUNK)) rounds.
_N_ROUNDS = math.ceil(math.log2(_CHUNK))
GATE_LOWER_BOUND = -5.0


def max_safe_subblock(gate_lower_bound=GATE_LOWER_BOUND, exp_max=88.7):
    """Largest usable (power-of-two) sub-block for a given gate lower bound."""
    n = int(exp_max / abs(gate_lower_bound))
    return 1 << (n.bit_length() - 1)


@nki.jit
def kda_cte(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    beta: nl.NkiTensor,
    gate: nl.NkiTensor,
    scale: float = 1.0,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """KDA chunked prefill.

    Args:
        q, k, v (nl.NkiTensor): ``[B, S, D]``. ``S`` must be a multiple of 64.
            ``q``/``k`` arrive **already l2-normed** (caller's job, as in gdn_cte).
        beta (nl.NkiTensor): ``[B, S]`` per-token update strength (post-sigmoid).
        gate (nl.NkiTensor): ``[B, S, D]`` **PER-CHANNEL** log-decay, negative.
            This is the KDA difference; GDN takes ``[B, S]``.
        scale (float): query scaling, typically ``1/sqrt(D)``.

    Returns:
        result ``[B, S, D]``, state_output ``[B, D, D]`` (float32).

    Note:
        ``B`` here is the flattened batch*head axis, one head per slice, matching
        gdn_cte's contract.
    """
    B, S, D = q.shape
    kernel_assert(S % _CHUNK == 0, f"kda_cte requires S % {_CHUNK} == 0, got S={S}")
    kernel_assert(D <= 128, f"D must fit the partition axis, got {D}")
    kernel_assert(
        gate.shape == (B, S, D),
        f"gate must be PER-CHANNEL [B, S, D], got {gate.shape}. A [B, S] gate is the "
        f"GDN form, not KDA.",
    )
    num_chunks = S // _CHUNK
    dtype = q.dtype

    result = nl.ndarray(shape=(B, S, D), dtype=dtype, buffer=nl.shared_hbm)
    state_output = nl.ndarray(shape=(B, D, D), dtype=nl.float32, buffer=nl.shared_hbm)

    # Cumsum operator, built once: U[k,i] = 1 if k <= i. nc_matmul contracts the
    # partition axis, so with tokens on partitions this yields an inclusive prefix sum
    # with tokens still on partitions. 0/1 so bf16 is exact; PSUM accumulates fp32.
    ones_CC = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(ones_CC, 1.0)
    U_cumsum = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
    nisa.affine_select(
        U_cumsum, pattern=[[1, _CHUNK]], offset=0, channel_multiplier=-1,
        cmp_op=nl.greater_equal, on_true_tile=ones_CC, on_false_value=0.0,
    )
    # Identity for the doubling solve: 1 where partition == free.
    Ident = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
    nisa.affine_select(
        Ident, pattern=[[-1, _CHUNK]], offset=0, channel_multiplier=1,
        cmp_op=nl.equal, on_true_tile=ones_CC, on_false_value=0.0,
    )

    for batch_id in nl.affine_range(B):
        state = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(state, 0.0)

        for chunk_idx in range(num_chunks):
            base = chunk_idx * _CHUNK

            # ---- loads: tokens on the partition axis ----
            k_sb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            v_sb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            q_sb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_sb, src=k[batch_id, nl.ds(base, _CHUNK), nl.ds(0, D)])
            nisa.dma_copy(dst=v_sb, src=v[batch_id, nl.ds(base, _CHUNK), nl.ds(0, D)])
            nisa.dma_copy(dst=q_sb, src=q[batch_id, nl.ds(base, _CHUNK), nl.ds(0, D)])
            beta_sb = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=beta_sb, src=beta[batch_id, nl.ds(base, _CHUNK)])

            # ---- per-channel cumulative gate: [CHUNK, D] ----
            gate_f32 = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gate_f32, src=gate[batch_id, nl.ds(base, _CHUNK), nl.ds(0, D)])
            gate_bf = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_bf, src=gate_f32, engine=nisa.vector_engine)
            cg_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(cg_p, U_cumsum, gate_bf)
            cg = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cg, src=cg_p, engine=nisa.vector_engine)

            # Transposed views: D on the partition axis. This is what makes the anchor a
            # [D, 1] per-partition operand, which is the whole reason the anchored form
            # is cheap here -- no partition broadcast is needed anywhere.
            cg_t = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            k_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            q_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=cg_t, src=cg)
            nisa.dma_transpose(dst=k_t, src=k_sb)
            nisa.dma_transpose(dst=q_t, src=q_sb)

            # k_beta in transposed layout: scale each token column by its beta.
            beta_row = nl.ndarray((1, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_transpose(dst=beta_row, src=beta_sb)
            kb_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=kb_t, data1=k_t, data2=beta_row, op=nl.multiply)

            # ================= CHANGE 1: anchored A, no [CHUNK, CHUNK, D] tensor ======
            A = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(A, 0.0)
            for bi in range(_N_SUBBLK):
                n = bi * _SUBBLK
                anchor = cg_t[:, nl.ds(n, 1)]                       # [D, 1] per-partition
                rows = nl.ds(n, _SUBBLK)
                # xs = k_beta * exp(cg - anchor)   (<= 1)
                e_row = nl.ndarray((D, _SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(e_row, cg_t[:, rows], op0=nl.subtract, operand0=anchor)
                nisa.activation(dst=e_row, data=e_row, op=nl.exp)
                xs = nl.ndarray((D, _SUBBLK), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=xs, data1=kb_t[:, rows], data2=e_row, op=nl.multiply)
                for bj in range(bi + 1):
                    cols = nl.ds(bj * _SUBBLK, _SUBBLK)
                    # ys = k * exp(anchor - cg)    (>= 1 on the diagonal block: the
                    # binding side, bounded by _SUBBLK)
                    e_col = nl.ndarray((D, _SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(e_col, cg_t[:, cols], op0=nl.multiply, operand0=-1.0)
                    nisa.tensor_scalar(e_col, e_col, op0=nl.add, operand0=anchor)
                    nisa.activation(dst=e_col, data=e_col, op=nl.exp)
                    ys = nl.ndarray((D, _SUBBLK), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=ys, data1=k_t[:, cols], data2=e_col, op=nl.multiply)
                    blk = nl.ndarray((_SUBBLK, _SUBBLK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(blk, xs, ys)                     # PLAIN MATMUL
                    nisa.tensor_scalar(
                        A[rows, cols], blk, op0=nl.multiply, operand0=-1.0
                    )
            # strictly lower triangular (j < i)
            A_masked = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.affine_select(
                A_masked, pattern=[[-1, _CHUNK]], offset=-1, channel_multiplier=1,
                cmp_op=nl.greater_equal, on_true_tile=A, on_false_value=0.0,
            )

            # ---- solve (I - A) M = I by recursive doubling; A is nilpotent ----
            # Track M^T: nc_matmul(stat, mov) = stat^T @ mov, so
            #   MT <- MT + nc_matmul(A, MT)      == (M (I + A))^T
            #   A  <- A @ A   via the pair below, as gdn_cte does
            A_bf = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=A_bf, src=A_masked, engine=nisa.vector_engine)
            A_t = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=A_t, src=A_bf)
            MT = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=MT, src=Ident, engine=nisa.scalar_engine)
            P_stat, P_t = A_bf, A_t
            for rd in range(_N_ROUNDS):
                pMT = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(pMT, P_stat, MT)
                MT_new = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=MT_new, data1=MT, data2=pMT, op=nl.add)
                MT = MT_new
                if rd < _N_ROUNDS - 1:
                    pp = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pp, P_stat, P_t)
                    ppt = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(ppt, P_t, P_stat)
                    P_t_new = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                    P_stat_new = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=P_t_new, src=pp, engine=nisa.scalar_engine)
                    nisa.tensor_copy(dst=P_stat_new, src=ppt, engine=nisa.scalar_engine)
                    P_t, P_stat = P_t_new, P_stat_new

            # ---- v_beta, k_cumdecay; both need exp(cg) applied BEFORE the matmul ----
            vb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(vb, v_sb, op0=nl.multiply, operand0=beta_sb,
                               engine=nisa.scalar_engine)
            exp_cg = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_cg, data=cg, op=nl.exp)
            kb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(kb, k_sb, op0=nl.multiply, operand0=beta_sb,
                               engine=nisa.scalar_engine)
            # CHANGE 3: elementwise per-channel FIRST, then the matmul.
            kb_exp = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=kb_exp, data1=kb, data2=exp_cg, op=nl.multiply)

            vnew_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(vnew_p, MT, vb)                       # = M @ v_beta
            kcd_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(kcd_p, MT, kb_exp)                    # = M @ (k_beta*exp_cg)

            state_bf = nl.ndarray((D, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=state_bf, src=state, engine=nisa.vector_engine)
            kcd_bf = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kcd_bf, src=kcd_p, engine=nisa.scalar_engine)
            kcd_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=kcd_t, src=kcd_bf)
            vprime = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(vprime, kcd_t, state_bf)              # k_cumdecay @ state
            # nisa.tensor_tensor cannot take BOTH operands from PSUM, so evict one.
            vnew_sb = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=vnew_sb, src=vnew_p, engine=nisa.vector_engine)
            v_new = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=v_new, data1=vnew_sb, data2=vprime, op=nl.subtract)
            v_new_bf = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_new_bf, src=v_new, engine=nisa.vector_engine)

            # ---- o_cross: CHANGE 3 again. (q * exp_cg) @ state, NOT (q @ state) * exp_cg
            q_scaled = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(q_scaled, q_sb, op0=nl.multiply, operand0=scale)
            nisa.tensor_tensor(dst=q_scaled, data1=q_scaled, data2=exp_cg, op=nl.multiply)
            qs_bf = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qs_bf, src=q_scaled, engine=nisa.vector_engine)
            qs_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=qs_t, src=qs_bf)
            o_cross = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(o_cross, qs_t, state_bf)

            # ================= CHANGE 2: anchored intra-chunk output ==================
            # Same bilinear form as A, with (q*scale) in place of k_beta.
            qsc_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(qsc_t, q_t, op0=nl.multiply, operand0=scale)
            QK = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(QK, 0.0)
            for bi in range(_N_SUBBLK):
                n = bi * _SUBBLK
                anchor = cg_t[:, nl.ds(n, 1)]
                rows = nl.ds(n, _SUBBLK)
                e_row = nl.ndarray((D, _SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(e_row, cg_t[:, rows], op0=nl.subtract, operand0=anchor)
                nisa.activation(dst=e_row, data=e_row, op=nl.exp)
                xs = nl.ndarray((D, _SUBBLK), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=xs, data1=qsc_t[:, rows], data2=e_row, op=nl.multiply)
                for bj in range(bi + 1):
                    cols = nl.ds(bj * _SUBBLK, _SUBBLK)
                    e_col = nl.ndarray((D, _SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(e_col, cg_t[:, cols], op0=nl.multiply, operand0=-1.0)
                    nisa.tensor_scalar(e_col, e_col, op0=nl.add, operand0=anchor)
                    nisa.activation(dst=e_col, data=e_col, op=nl.exp)
                    ys = nl.ndarray((D, _SUBBLK), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=ys, data1=k_t[:, cols], data2=e_col, op=nl.multiply)
                    blk = nl.ndarray((_SUBBLK, _SUBBLK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(blk, xs, ys)
                    nisa.tensor_copy(dst=QK[rows, cols], src=blk, engine=nisa.vector_engine)
            # NOTE the offset difference from A_masked above, and it is not cosmetic:
            # A keeps j < i STRICTLY (offset=-1), the intra output keeps j <= i
            # INCLUSIVE (offset=0). chunk_kda spells these as masked_fill(tri) vs
            # masked_fill(stri); gdn_cte as offset=-1 vs offset=0. Using -1 here drops
            # the diagonal term, which on chunk 0 (state still zero) makes row 0
            # exactly zero -- 99.8% output error with the state still correct.
            QK_masked = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.affine_select(
                QK_masked, pattern=[[-1, _CHUNK]], offset=0, channel_multiplier=1,
                cmp_op=nl.greater_equal, on_true_tile=QK, on_false_value=0.0,
            )
            QK_bf = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=QK_bf, src=QK_masked, engine=nisa.vector_engine)
            QK_t = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=QK_t, src=QK_bf)
            o_intra = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(o_intra, QK_t, v_new_bf)

            # same PSUM restriction as v_new above
            o_cross_sb = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_cross_sb, src=o_cross, engine=nisa.vector_engine)
            o_out = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o_out, data1=o_cross_sb, data2=o_intra, op=nl.add)
            nisa.dma_copy(dst=result[batch_id, nl.ds(base, _CHUNK), nl.ds(0, D)], src=o_out)

            # ---- state update ----
            # CHANGE 4: exp(cg_last) is a genuine [D, 1] per-partition column. Free.
            cg_last = cg_t[:, nl.ds(_CHUNK - 1, 1)]                     # [D, 1]
            exp_last = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_last, data=cg_last, op=nl.exp)
            state_dec = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(state_dec, state, op0=nl.multiply, operand0=exp_last,
                               engine=nisa.scalar_engine)
            # k_dec = k * exp(cg_last - cg), per channel, <= 1 and benign
            dte = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dte, cg_t, op0=nl.multiply, operand0=-1.0)
            nisa.tensor_scalar(dte, dte, op0=nl.add, operand0=cg_last)
            nisa.activation(dst=dte, data=dte, op=nl.exp)
            k_dec_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=k_dec_t, data1=k_t, data2=dte, op=nl.multiply)
            k_dec = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=k_dec, src=k_dec_t)
            su = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(su, k_dec, v_new_bf)                      # k_dec^T @ v_new
            state_new = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=state_new, data1=su, data2=state_dec, op=nl.add)
            nisa.tensor_copy(dst=state, src=state_new, engine=nisa.vector_engine)

        nisa.dma_copy(dst=state_output[batch_id, nl.ds(0, D), nl.ds(0, D)], src=state)

    return result, state_output
