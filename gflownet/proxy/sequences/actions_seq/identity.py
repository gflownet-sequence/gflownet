"""
Action sequence objective function, defined for unbounded box-like environments,
mainly for the movements in a 2D grid, where we score the sequences of actions
that we take.

The function places high scores along the identity action sequence, i.e.
sticking to the line y = x. Anywhere that's near the line i.e. y = x - 1 or
y = x + 1 will be scored lower, and anywhere further away from the line will be
scored even lower.
"""

from typing import Iterable

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class ActionsIdentity(Proxy):
    """
    It is assumed that the state values will be a sequence of actions,
    either: U, R, L, D, where U is up, R is right, L is left and D is down.
    """

    def __init__(
        self,
        scores: Iterable = [2.0, 0.5, 0.01],
        **kwargs,
    ):
        """
        Initializes an instance of the identity proxy.

        Parameters
        ----------
        scores : iterable
            The scores of the three regions for the non-Gaussian version. It should be
            a list with three elements, where the first element is highest score, the
            second element the middle score and the third element the lowest score.
        """
        super().__init__(**kwargs)
        self.scores = tuple(scores)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        # I think we're expecting a list of lists of actions
        rewards = torch.zeros(len(states), device=self.device, dtype=self.float)
        for i, state in enumerate(states):
            # we count all the actions in the sequence
            y_count = state.count("U") - state.count("D")
            x_count = state.count("R") - state.count("L")

            # we then calculate the distance from the identity line
            distance = abs(y_count - x_count)

            # we then score the state based on its distance from the identity line
            if distance == 0:
                rewards[i] = self.scores[0]
            elif distance == 1:
                rewards[i] = self.scores[1]
            else:
                rewards[i] = self.scores[2]
        return rewards
