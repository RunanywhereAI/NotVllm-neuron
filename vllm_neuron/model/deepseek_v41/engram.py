# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hashing, on the host.

The hash multiplies token ids by multipliers up to ``int64_max / vocab / 2`` and XORs
the products, so it needs exact 64-bit integer arithmetic. Neuron computes integer
products through float32 on several engines, exact only below 2^24. So the runner
computes the hash rows on the CPU from the request's token history and hands them
to the model as ``engram_ids``.

This reproduces ``ref/engram.py`` (``EngramLayout``, ``compute_hash_multipliers``,
``NgramHashState``) without sympy or a live tokenizer. The tokenizer-derived
compressed-token map is computed once and passed in. Image spans (DEAD tokens) are
out of scope: text only.
"""

from __future__ import annotations

import numpy as np
import torch


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def bucket_primes(layer_ids, max_ngram_size: int, n_heads: int, vocab_size: int):
    """``[layer][ngram-1][head]`` moduli: primes above ``vocab_size - 1``, drawn in order and
    never reused, as ``EngramLayout.from_args``."""
    primes, seen = [], set()
    for _ in layer_ids:
        per = []
        for _ in range(max_ngram_size - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per.append(sizes)
        primes.append(per)
    return primes


def hash_multipliers(layer_ids, max_ngram_size: int, compressed_vocab: int) -> np.ndarray:
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = [np.random.default_rng(10007 * lid).integers(0, bound, size=(max_ngram_size,),
                                                        dtype=np.int64) * 2 + 1
            for lid in layer_ids]
    return np.stack(rows)


class EngramHasher:
    """Token history -> hash rows ``[T, n_engram_layers, (max_ngram-1) * n_heads]``."""

    def __init__(self, args, token_map, compressed_vocab: int):
        if compressed_vocab != args.engram_compressed_vocab_size:
            raise ValueError(f"compressed vocab {compressed_vocab} != config "
                             f"{args.engram_compressed_vocab_size}: every multiplier derives from it")
        self.n = args.engram_max_ngram_size
        layers = tuple(args.engram_layer_ids)
        primes = bucket_primes(layers, self.n, args.engram_n_heads, args.engram_vocab_size)
        flat = [[p for per in layer for p in per] for layer in primes]
        self.primes = torch.tensor(primes, dtype=torch.int64)                # [L, n-1, H]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *f[:-1]]) for f in flat]))
        self.mult = torch.from_numpy(hash_multipliers(layers, self.n, compressed_vocab))
        self.token_map = torch.as_tensor(token_map, dtype=torch.int64)
        self.pad = int(self.token_map[args.engram_pad_id])
        for lid, rows, f in zip(layers, args.engram_num_embeddings, flat):
            if sum(f) > rows:
                raise ValueError(f"engram layer {lid}: buckets need {sum(f)} rows, table has {rows}")

    def __call__(self, tokens, start: int, count: int) -> torch.Tensor:
        """Hash rows for positions ``start .. start+count-1`` of the sequence ``tokens``
        (raw ids, at least ``start + count`` long)."""
        ids = torch.as_tensor(tokens, dtype=torch.int64)
        pos = torch.arange(start, start + count)
        window = ids[(pos.unsqueeze(1) - torch.arange(self.n)).clamp_min(0)]
        return self.hash_windows(window, pos)

    def hash_windows(self, window: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``window [T, n]``: raw ids at positions ``p, p-1, .., p-n+1`` (any value where
        that is negative); ``pos [T]``: ``p``. -> ``[T, n_layers, cols]`` int64."""
        ids = self.token_map[window.to(torch.int64)]
        cols, blocked = [], torch.zeros(pos.shape[0], dtype=torch.bool)
        for shift in range(self.n):
            blocked = blocked | (pos < shift)
            cols.append(torch.where(blocked, torch.full_like(ids[:, shift], self.pad), ids[:, shift]))
        toks = torch.stack(cols, dim=-1)                                     # [T, n]
        prod = toks.unsqueeze(1) * self.mult                                 # [T, L, n]
        rolling, out = prod[..., 0], []
        for i in range(1, self.n):
            rolling = torch.bitwise_xor(rolling, prod[..., i])
            out.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(out, dim=-1) + self.offsets


def compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """The reference's ``build_compressed_token_map``: tokens that normalise alike share an
    id, and the number of distinct ids seeds every hash multiplier. Needs the HF fast
    tokenizer (its Rust backend decodes without clean-up, as training did)."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


def hasher_for(args, model_dir) -> EngramHasher:
    """The hasher for a checkpoint directory. Builds the token map from the tokenizer,
    once per rank: a few seconds over 129k tokens."""
    from transformers import AutoTokenizer

    lookup, vocab = compressed_token_map(AutoTokenizer.from_pretrained(str(model_dir)))
    return EngramHasher(args, lookup, vocab)
