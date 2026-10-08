"""Tensor/expert parallelism: N gloo ranks must reproduce the oracle.

Each rank is its own process, builds the model with its ``Parallel`` coordinates, loads
only its slices (``local_tensor``), and runs the Engram scenario of
``test_plugin_engram.py`` (padded prefill, a prefix hit, interleaved batched decode with
pad rows) with every collective real. Its gathered logits must match the oracle.

TP=8 on the 8-head test config leaves one head per rank, half of an o_group -- the path the released model takes at TP=64 (1 head of 8 per
group). A sharded run can match for the wrong reason, so each sharded parameter is
also loaded from the WRONG rank's slice on one rank, and the comparison must fail.
"""

import functools
import os
import pathlib
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

ENGRAM = dict(engram_layer_ids=(1, 4), engram_num_embeddings=(400, 600), engram_max_ngram_size=4,
              engram_vocab_size=50, engram_n_heads=2, engram_head_dim=32, engram_pad_id=2,
              engram_compressed_vocab_size=300)
TOKEN_MAP = [t % 300 for t in range(512)]


class Gloo:
    """The slice of vLLM's GroupCoordinator the model uses. Reduces in fp32: gloo's bf16
    support varies, and every bf16 reduction here sums one real value with zeros."""

    def __init__(self, world):
        self.world = world

    def all_reduce(self, t):
        out = t.float().clone()
        dist.all_reduce(out)
        return out.to(t.dtype)

    def all_gather(self, t, dim):
        parts = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=dim)


def _reference():
    from personal_reference.deepseek_v41 import oracle
    from personal_reference.deepseek_v41.tests import harness

    oracle.load_reference("exact")
    sys.modules["engram"].build_compressed_token_map = lambda tok: (TOKEN_MAP, 300)
    return harness.build("exact", 0, **ENGRAM)


def _scenario(model, args):
    from personal_reference.deepseek_v41.tests import plugin_harness as ph

    E = ph.import_plugin("vllm_neuron.model.deepseek_v41.engram")
    hasher = E.EngramHasher(args, TOKEN_MAP, 300)
    torch.manual_seed(1)
    A = torch.randint(0, args.vocab_size, (60,)).tolist()
    B = A[:32] + torch.randint(0, args.vocab_size, (28,)).tolist()
    with torch.no_grad():
        run = ph.FakeRunner(model)
        la = [run.prefill("a", A[:37], 0, bucket=48, engram_ids=hasher(A, 0, 37))]
        run.share_prefix("a", "b", 32)
        lb = [run.prefill("b", B[32:45], 32, bucket=16, engram_ids=hasher(B, 32, 13))]
        for pa, pb in zip(range(37, 50), range(45, 58)):
            e = torch.cat([hasher(A, pa, 1), hasher(B, pb, 1)])
            out = run.decode(["a", "b"], [A[pa], B[pb]], [pa, pb], pad_rows=1, engram_ids=e)
            la.append(out[:1])
            lb.append(out[1:])
    return torch.cat(la), torch.cat(lb), A, B


def _worker(rank, world, ep, port, out_dir, sabotage):
    torch.set_num_threads(1)
    from personal_reference.deepseek_v41.tests import plugin_harness as ph

    M = ph.import_plugin("vllm_neuron.model.deepseek_v41.model")
    # before joining the group: the reference shards itself when dist is initialised
    ref, args, _ = _reference()
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    etp = world // ep

    def coords(r):
        return M.Parallel(tp=world, rank=r, ep=ep, ep_rank=r // etp, etp=etp, etp_rank=r % etp,
                          group=Gloo(world))

    model = M.DeepseekV41Model(args, block_size=8, cache_dtype=torch.float32,
                               par=coords(rank)).set_dtype(torch.float32)
    src = M.DictSource(ph.dequantized_state_dict(ref))
    if sabotage and rank == 1:
        with torch.device("meta"):
            other = M.DeepseekV41Model(args, block_size=8, cache_dtype=torch.float32,
                                       par=coords((rank + 1) % world))
        real = model.local_tensor

        def local_tensor(name, s):
            return other.local_tensor(name, s) if name == sabotage else real(name, s)

        model.local_tensor = local_tensor
    model.load_from(src)
    la, lb, _, _ = _scenario(model.eval(), args)
    torch.save({"a": la, "b": lb}, os.path.join(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _oracle(seq):
    ref, _, _ = _reference()
    ref.head.forward = functools.partial(type(ref.head).forward, ref.head, full_logits=True)
    with torch.no_grad():
        _, logits, _ = ref(torch.tensor([seq]), 0)
    return logits[0]


@pytest.fixture(scope="module")
def oracle_logits():
    _, args, _ = _reference()
    torch.manual_seed(1)
    A = torch.randint(0, args.vocab_size, (60,)).tolist()
    B = A[:32] + torch.randint(0, args.vocab_size, (28,)).tolist()
    return _oracle(A[:50]), _oracle(B[:58])[32:58]


def _run(world, ep=None, sabotage=None, port=29511):
    ep = world if ep is None else ep
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_worker, args=(world, ep, port, d, sabotage), nprocs=world, join=True)
        return [torch.load(os.path.join(d, f"rank{r}.pt")) for r in range(world)]


def _err(outs, oracle_logits):
    OA, OB = oracle_logits

    def rel(a, b):
        return ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()

    return max(max(rel(o["a"], OA), rel(o["b"], OB)) for o in outs)


@pytest.mark.parametrize("world,ep", [(2, 2), (4, 4), (8, 8), (8, 4)])
def test_parallel_matches_oracle(oracle_logits, world, ep):
    err = _err(_run(world, ep, port=29500 + world * 10 + ep), oracle_logits)
    assert err < 1e-5, err


@pytest.mark.parametrize("name", [
    "layers.2.attn.wq_b.weight", "layers.2.attn.attn_sink", "layers.2.attn.wo_a.weight",
    "layers.2.attn.wo_b.weight", "layers.2.ffn.gate_up_proj", "layers.2.ffn.down_proj",
    "layers.2.ffn.shared_experts.w1.weight", "layers.1.engram.embed.weight",
    "layers.4.engram.wkv.weight", "embed.weight", "head.weight",
])
def test_a_wrong_shard_is_caught(oracle_logits, name):
    err = _err(_run(8, 4, sabotage=name, port=29600 + hash(name) % 300), oracle_logits)
    assert err > 1e-3, f"{name} loaded from the wrong rank went unnoticed ({err})"
