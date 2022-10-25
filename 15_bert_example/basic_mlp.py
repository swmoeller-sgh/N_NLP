from typing import Any, Union

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class MLP(nn.Module):
    """Multi-layer-perceptron implementation with variable number of hidden layers and a linear output layer."""

    def __init__(
        self,
        hidd_size: int,
        num_layers: int,
        input_size: int = 2,
        output_size: int = 1,
        spectral_normalization: bool = False,
    ):
        """Instantiate MLP model.

        Args:
            input_size (:obj:`int`): number of input observed_features.
            hidd_size (:obj:`int`): number of hidden observed_features.
            num_layers (:obj:`int`): number of hidden layers.
            output_size (:obj:`int`): number of output observed_features.
            spectral_normalization: if True, spectral normalization is applied to all linear layers
        """
        super(MLP, self).__init__()
        self.spectral_normalization = spectral_normalization
        self.num_layers = num_layers
        self.relu = nn.CELU()
        if spectral_normalization:
            self.input = spectral_norm(nn.Linear(input_size, hidd_size))
            self.hidd = nn.ModuleList(
                [nn.Sequential(spectral_norm(nn.Linear(hidd_size, hidd_size)), self.relu) for _ in range(num_layers)]
            )
            self.output = spectral_norm(nn.Linear(hidd_size, output_size))
        else:
            self.input = nn.Linear(input_size, hidd_size)
            self.hidd = nn.ModuleList(
                [nn.Sequential(nn.Linear(hidd_size, hidd_size), self.relu) for _ in range(num_layers)]
            )
            self.output = nn.Linear(hidd_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MLP output.

        Args:
            x (:obj:`torch.Tensor`): input tensor of shape (batchsize, input_size)

        Returns:
            (:obj:`torch.Tensor`): output tensor of shape (batchsize, output_size)
        """
        h = self.relu(self.input(x))
        for layer in self.hidd:
            h = layer(h)

        output = self.output(h)
        return output
