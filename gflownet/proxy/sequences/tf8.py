"""
Reward for the TF-8 Binding Task (SIX6_REF_R1 landscape)

Data and protocol from sci-modeling-bench:
https://github.com/xukp20/sci-modeling-bench/tree/main/docs

Two modes:
  - "offline" (default, the benchmark setting): fit a ridge model on the
    bottom-50% sequences only, and use its predictions as the reward.
    The hidden top half is never read.
  - "exact": look up the true score of every sequence in the full table.
    Not the benchmark setting -- use it to sanity-check your GFlowNet
    sampler, and say so if you report results with it.

Final evaluation should always go through task.evaluate() (see
evaluate_tfbind8 below), not through this proxy.
"""

import logging

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

# Original v1.0.0 data release, as pinned in the sci-modeling-bench docs.
REVISION = "2ee2856f4255bb6a64c11b6c2660a6f41418e654"
BASES = "ACGT"
# Token order of the DNA environment (gflownet/envs/sequences/dna.py):
# token 1=A, 2=C, 3=T, 4=G, and 0 = padding. Note T and G are NOT in
# alphabetical order there, so tokens are remapped before the lookup.
ENV_TOKENS = ("A", "C", "T", "G")
SEQ_LEN = 8
N_SEQS = 4**SEQ_LEN  # 65,536

log = logging.getLogger(__name__)
for _name in ("httpx", "httpx2", "huggingface_hub", "datasets"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def seq_to_index(seq: str) -> int:
    """Base-4 index of a DNA 8-mer, with A=0, C=1, G=2, T=3."""
    idx = 0
    for c in seq:
        idx = idx * 4 + BASES.index(c)
    return idx


def all_onehot() -> torch.Tensor:
    """One-hot features (65536, 32) for every 8-mer, ordered by seq_to_index."""
    pows = 4 ** torch.arange(SEQ_LEN - 1, -1, -1)
    digits = (torch.arange(N_SEQS)[:, None] // pows) % 4  # (65536, 8)
    return torch.nn.functional.one_hot(digits, 4).reshape(N_SEQS, -1).double()


class TFBind8Scorer(Proxy):
    def __init__(
        self,
        mode: str = "offline",
        ridge_alpha: float = 1.0,
        revision: str = REVISION,
        tokens: tuple = ENV_TOKENS,
        log_every: int = 50,
        top_k: int = 100,
        max_log_batch: int = 256,
        **kwargs,
    ):
        """
        Parameters
        ----------
        mode : str
            "offline" (ridge proxy trained on the bottom 50%) or "exact"
            (true lookup over all 65,536 sequences).
        ridge_alpha : float
            Ridge regularization strength for "offline" mode. The
            sci-modeling-bench reference baseline uses 1.0.
        revision : str
            Hugging Face dataset commit to load.
        tokens : tuple
            Token order used by the environment (token i+1 = tokens[i]).
            Must match NUCLEOBASES in the DNA environment.
        log_every : int
            Write a progress line to the log every this many training batches
            (0 turns it off).
        top_k : int
            In "exact" mode, track how many of the true top-k sequences have
            been sampled so far.
        max_log_batch : int
            Batches larger than this (the 1000-sequence test set, evaluator
            runs) are scored but not counted in the progress line.
        """
        super().__init__(**kwargs)
        from sci_modeling_bench.suites.design_bench import (
            TFBind8Dataset,
            TFBind8DesignBenchProtocol,
        )

        dataset = TFBind8Dataset.from_hub(revision=revision)

        if mode == "offline":
            data = TFBind8DesignBenchProtocol().build_input(dataset).data
            table = self._fit_ridge(
                list(data["sequence"]), list(data["normalized_e_score"]), ridge_alpha
            )
        elif mode == "exact":
            protocol = TFBind8DesignBenchProtocol(min_percentile=0.0, max_percentile=100.0)
            data = protocol.build_input(dataset).data
            table = torch.full((N_SEQS,), float("nan"), dtype=torch.float64)
            for seq, y in zip(data["sequence"], data["normalized_e_score"]):
                table[seq_to_index(seq)] = y
            assert not torch.isnan(table).any(), "Full table is missing sequences"
        else:
            raise ValueError(f"mode must be 'offline' or 'exact', got {mode!r}")

        self.mode = mode
        # One score per possible 8-mer, so __call__ is a pure lookup.
        self.table = table.to(device=self.device, dtype=self.float)
        self.pows = 4 ** torch.arange(SEQ_LEN - 1, -1, -1, device=self.device)
        tokens = tuple(tokens)
        assert sorted(tokens) == list(BASES), f"tokens must be a permutation of ACGT, got {tokens}"
        # env token (0..4) -> base index in A=0, C=1, G=2, T=3 (padding -> 0, masked later)
        self.token2base = torch.tensor(
            [0] + [BASES.index(t) for t in tokens], device=self.device
        )
        self.nucleotide_map = {0: "", **{i + 1: t for i, t in enumerate(tokens)}}

        # Progress tracking
        self.log_every = log_every
        self.max_log_batch = max_log_batch
        self._n_batches = 0
        self._window = []  # scores since the last progress line
        self._best = float("-inf")
        self._seen = torch.zeros(N_SEQS, dtype=torch.bool, device=self.device)
        # True top-k only exists in exact mode (offline must not read it)
        self.top_k = top_k if mode == "exact" else 0
        if self.top_k:
            self._top_mask = torch.zeros(N_SEQS, dtype=torch.bool, device=self.device)
            self._top_mask[self.table.topk(top_k).indices] = True

    @staticmethod
    def _fit_ridge(seqs, ys, alpha) -> torch.Tensor:
        """Closed-form ridge on position-wise one-hot features.

        Trains on the offline sequences only, then predicts all 65,536.
        """
        feats = torch.cat([all_onehot(), torch.ones(N_SEQS, 1, dtype=torch.float64)], 1)
        idx = torch.tensor([seq_to_index(s) for s in seqs])
        X = feats[idx]
        y = torch.tensor(ys, dtype=torch.float64)
        reg = alpha * torch.eye(X.shape[1], dtype=torch.float64)
        reg[-1, -1] = 0.0  # don't shrink the bias
        w = torch.linalg.solve(X.T @ X + reg, X.T @ y)
        return feats @ w

    def states_to_tokens(self, states) -> torch.Tensor:
        """(batch, seq_len * 5) one-hot states -> (batch, seq_len) tokens in 0..4."""
        states = torch.as_tensor(states, device=self.device)
        tokens = states.reshape(states.shape[0], -1, 5).argmax(dim=-1)
        if tokens.shape[1] != SEQ_LEN:
            raise ValueError(
                f"Expected sequences of length {SEQ_LEN}, got {tokens.shape[1]}. "
                "Set the environment's max length to 8."
            )
        return tokens

    def states_to_strings(self, states) -> list[str]:
        tokens = self.states_to_tokens(states).tolist()
        return ["".join(self.nucleotide_map[t] for t in row) for row in tokens]

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        tokens = self.states_to_tokens(states)
        complete = (tokens > 0).all(dim=1)  # token 0 = empty position
        idx = (self.token2base[tokens] * self.pows).sum(dim=1)
        scores = self.table[idx]
        # Incomplete sequences have no score; give them the minimum.
        scores = torch.where(complete, scores, torch.zeros_like(scores))
        if self.log_every and 0 < len(scores) <= self.max_log_batch:
            self._track(idx[complete], scores[complete])
        return scores

    def _track(self, idx, scores):
        self._n_batches += 1
        self._seen[idx] = True
        self._window.append(scores.detach())
        if self._n_batches % self.log_every:
            return
        w = torch.cat(self._window)
        self._window = []
        self._best = max(self._best, w.max().item()) if len(w) else self._best
        msg = (
            f"[tf8] batch {self._n_batches} | mean score {w.mean().item():.3f} "
            f"| max {w.max().item():.3f} | best so far {self._best:.3f} "
            f"| unique seqs {int(self._seen.sum())}"
        )
        if self.top_k:
            found = int((self._seen & self._top_mask).sum())
            msg += f" | top-{self.top_k} found {found}/{self.top_k}"
        log.info(msg)


def evaluate_tfbind8(sequences: list[str], revision: str = REVISION):
    """Official sci-modeling-bench score for 32 distinct sequences.

    Pass them ordered from your model's best to worst guess.
    """
    from sci_modeling_bench.suites.design_bench import TFBind8BlackBoxOptimizationTask

    task = TFBind8BlackBoxOptimizationTask.from_hub(revision=revision)
    result = task.evaluate([{"sequence": s} for s in sequences])
    if not result.submission_valid:
        print("Submission invalid -- need exactly 32 distinct valid 8-mers.")
    return result