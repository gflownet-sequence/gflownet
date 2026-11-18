"""
Class to represent movements as a sequence of actions.

e.g.: we always start from (0, 0), then if we have URR it corresponds to
moving up then right and right, so we would get (2, 1).

We'll assume a 2D grid example, where we can move only up or right.
"""

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from torchtyping import TensorType
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gflownet.envs.sequences.base import SequenceBase
from gflownet.utils.common import tlong

ACTIONS = ("U", "R", "L", "D")
PAD_TOKEN = "EOS"


class ActionsSeq(SequenceBase):
    """
    A DNA sequence environment, where the tokens are the nucleobases A, C, T and G.

    This basic implementation fully relies on the base sequence class Sequence.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        We will have no specific proxy format for this environment.
        """
        super().__init__(tokens=ACTIONS, pad_token=PAD_TOKEN, **kwargs)

    def plot_reward_samples(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        samples_reward: TensorType["batch_size", "state_proxy_dim"],
        rewards: TensorType["batch_size"],
        dpi: int = 150,
        n_ticks_max: int = 50,
        reward_norm: bool = True,
        **kwargs,
    ):
        """
        Plots the reward density as a 2D histogram on the grid, alongside a histogram
        representing the samples density.

        It is assumed that the rewards correspond to entire domain of the grid and are
        sorted from left to right (first) and top to bottom of the grid of samples.

        Parameters
        ----------
        samples : tensor
            A batch of samples from the GFlowNet policy in proxy format. These samples
            will be plotted on top of the reward density.
        samples_reward : tensor
            A batch of samples containing a grid over the sample space, from which the
            reward has been obtained. Ignored by this method.
        rewards : tensor
            The rewards of samples_reward. It should be a vector of dimensionality
            length ** 2 and be sorted such that the each block at rewards[i *
            length:i * length + length] correspond to the rewards at the i-th
            row of the grid of samples, from top to bottom.
        dpi : int
            Dots per inch, indicating the resolution of the plot.
        n_ticks_max : int
            Maximum of number of ticks to include in the axes.
        reward_norm : bool
            Whether to normalize the histogram. True by default.
        """

        # convert the tensor samples into a list of strings
        sample_strs = self.states2proxy(samples)

        # now let's create a function that turns a sample into a 2D coordinate
        def sample_to_coord(sample: str):
            """
            Converts a sample of actions into a 2D coordinate.
            """
            x = sample.count("R") - sample.count("L")
            y = sample.count("U") - sample.count("D")
            return x, y

        # let's figure out the min/max x and min/max y coordinates we can get from the samples
        coords = np.array([sample_to_coord(sample) for sample in sample_strs])

        min_x = coords[:, 0].min()
        max_x = coords[:, 0].max()
        min_y = coords[:, 1].min()
        max_y = coords[:, 1].max()

        x_len = max_x - min_x
        y_len = max_y - min_y

        # Init figure
        fig, axes = plt.subplots(ncols=2, dpi=dpi)
        xstep_ticks = np.ceil(x_len / n_ticks_max).astype(int)
        ystep_ticks = np.ceil(y_len / n_ticks_max).astype(int)

        # 2D histogram of samples
        samples_hist, xedges, yedges = np.histogram2d(
            coords[:, 0], coords[:, 1], bins=(x_len, y_len), density=True
        )
        samples_hist = samples_hist.T[::-1, :]
        # Normalize and reshape reward into a grid with [0, 0] at the bottom left
        if reward_norm:
            rewards = rewards / rewards.sum()
        rewards_2d = rewards.reshape(x_len, y_len).T[::-1, :]

        # Plot reward
        self._plot_grid_2d(rewards_2d, axes[0], min_x, min_y, xstep_ticks, ystep_ticks, title="True reward")
        # Plot samples histogram
        self._plot_grid_2d(samples_hist, axes[1], min_x, min_y, xstep_ticks, ystep_ticks, title="Samples density")
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_grid_2d(img: np.array, ax: Axes, xmin: int, ymin: int, xstep_ticks: int, ystep_ticks: int, title: str):
        """
        Plots a 2D histogram of a grid environment as an image.

        Parameters
        ----------
        img : np.array
            An array containing a 2D histogram over a grid.
        ax : Axes
            A matplotlib Axes object on which the image will be plotted.
        step_ticks : int
            The step value to add ticks to the axes. For example, if it is 2, the ticks
            will be at 0, 2, 4, ...
        title : str
            Title for the axes.
        """
        ax_img = ax.imshow(img)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        ax.set_xticks(np.arange(start=xmin, stop=img.shape[0], step=xstep_ticks))
        ax.set_yticks(np.arange(start=ymin, stop=img.shape[1], step=ystep_ticks)[::-1])
        cax.set_title(title)
        plt.colorbar(ax_img, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
