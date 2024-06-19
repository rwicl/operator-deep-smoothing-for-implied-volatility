from typing import Sequence

import torch
from torch import Tensor
from torch.nn import Module

from op_ds.utils.fnn import FNN


class FNNKernelTransform(Module):
    def __init__(self, channels: int, spatial_dim: int, hidden_channels: Sequence[int], **kwargs) -> None:
        super().__init__()

        self.fnn = FNN.from_config((2 * spatial_dim, *hidden_channels, channels * (channels + 1)), **kwargs)
        self.channels = channels
        self.spatial_dim = spatial_dim

    def forward(self, pos_i: Tensor, pos_j: Tensor, v_j: Tensor, **kwargs) -> Tensor:
        pos_ij = torch.cat((pos_i, pos_j), dim=-1)
        kernel = self.fnn(pos_ij).reshape(-1, self.channels, self.channels + 1)
        kernel, bias = kernel[..., :, :-1], kernel[..., :, -1]
        return torch.einsum(
            '...ij,...j->...i',
            kernel,
            v_j
        ) + bias


class NonlinearKernelTransformWithSkip(Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, spatial_dim: int, hidden_channels: Sequence[int], **kwargs) -> None:
        super().__init__()

        self.fnn = FNN.from_config((2 * spatial_dim + in_channels + skip_channels, *hidden_channels, out_channels * (in_channels + 1)), **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.spatial_dim = spatial_dim

    def forward(self, pos_i: Tensor, pos_j: Tensor, v_j: Tensor, x_j: Tensor, **kwargs) -> Tensor:
        pos_ij = torch.cat((pos_i, pos_j, v_j, x_j), dim=-1)
        kernel = self.fnn(pos_ij).reshape(-1, self.out_channels, self.in_channels + 1)
        kernel, bias = kernel[..., :, :-1], kernel[..., :, -1]
        return torch.einsum(
            '...ij,...j->...i',
            kernel,
            v_j
        ) + bias