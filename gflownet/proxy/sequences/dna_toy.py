"""
Reward for the Toy DNA Sequence scorer environment.

This scores things according to the weight in the nucleotide score dictionary.
Naturally, we will likely want to have all of the highest scoring nucleotides
will likely be the ones that are the result of the reward function itself.
"""

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class DNAToyScorer(Proxy):
    def __init__(
        self,
        nucleotide_scores: dict[str, float],
        **kwargs,
    ):
        """
        Parameters
        ----------
        nucleotide_scores : dict
            The score for each nucleotide.
        """
        super().__init__(**kwargs)
        self.nucleotide_scores = nucleotide_scores

        nucleotide_score_array = [0]
        nucleotide_score_array.extend([nucleotide_scores[nuc] for nuc in ["A", "C", "G", "T"]])

        self.nucleotide_score_tensor = tfloat(
            nucleotide_score_array,
            float_type=self.float,
            device=self.device,
        )

        self.nucleotide_map = {1: "A", 2: "C", 3: "G", 4: "T", 0: ""}

    def score_sequence(self, state_seq: TensorType["seq_len", 5]) -> float:
        """
        Score the sequence according to the nucleotide scores.
        """
        # we can take the sum across the rows
        # and then do the dot product with the nucleotide scores
        row_sums = state_seq.sum(dim=0)
        assert row_sums.shape[0] == 5, "The row sums should have shape (5,)"
        return row_sums.dot(self.nucleotide_score_tensor).item()

    def __call__(self, states: TensorType["batch", 1]) -> TensorType["batch"]:
        states = torch.as_tensor(states, device=self.device, dtype=self.float)

        rewards = torch.zeros(states.shape[0], device=self.device, dtype=self.float)

        for i in range(states.shape[0]):
            # reshape each one into 2D tensor of shape(n, 5) where n is the number of nucleotides in the sequence
            current_seq = states[i].reshape(-1, 5)
            # then we interpret the current seq
            rewards[i] = self.score_sequence(current_seq)

        return rewards
