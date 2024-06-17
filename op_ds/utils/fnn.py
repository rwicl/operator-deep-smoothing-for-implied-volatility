""" Basic feedforward neural network components."""

from typing import Sequence, Optional

import torch
from torch import Tensor
from torch.nn import Module, Sequential
import torch.nn.functional as F


def _module_lookup(module: str | Module) -> Module:
    """ Look up module by name or return module if already a module.
    """
    modules = torch.nn.__dict__.keys()
    module_dict = dict(zip(map(str.lower, modules), modules))

    if isinstance(module, str):
        return getattr(torch.nn, module_dict[module.lower()])()
    else:
        return module


class Linear(Module):
    """ Linear layer with optional spatial dimension.

    The spatial dimension determines the number of axes (on the right) along which to operate pointwise.

    """
    def __init__(self, in_channels: int, out_channels: int, spatial_dim: int = 0, bias: bool = True) -> None:
        super().__init__()

        if spatial_dim == 0:
            self._linear = torch.nn.Linear(in_channels, out_channels, bias=bias)
            self._forward = F.linear
        else:
            self._linear = getattr(torch.nn, f"Conv{spatial_dim}d")(in_channels, out_channels, kernel_size=1, bias=bias)
            self._forward = getattr(F, f"conv{spatial_dim}d")
        
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        return self._forward(input, self.weight, self.bias)

    @property
    def weight(self) -> Tensor:
        return self._linear.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self._linear.bias

    def __repr__(self) -> str:
        return f"{self._get_name()}({self.in_channels}, {self.out_channels}, spatial_dim={self.spatial_dim}, bias={self.bias is not None})"


class FNNLayer(Sequential):
    """ Feedforward neural network layer.
    """
    def __init__(self, linear: Linear, batch_norm: Optional[Module] = None,
                  activation: Optional[Module | str] = None, dropout: float = 0.) -> None:
        super().__init__()

        if batch_norm is not None:
            self.add_module('batch_norm', batch_norm)
        else:
            self.register_parameter('batch_norm', None)
        
        self.add_module('linear', linear)
        
        if isinstance(activation, str):
            activation = _module_lookup(activation)

        if activation is not None:
            self.add_module('activation', activation)
        else:
            self.register_parameter('activation', None)

        if dropout > 0.:
            self.add_module('dropout', torch.nn.Dropout(dropout))
        else:
            self.register_parameter('dropout', None)

    @classmethod
    def from_config(cls, in_channels: int, out_channels: int, spatial_dim: int = 0, 
                    batch_norm: bool = False, **kwargs) -> 'FNNLayer':

        linear = Linear(in_channels, out_channels, spatial_dim=spatial_dim, bias=kwargs.get('bias', True))
        activation = kwargs.get('activation', None)
        dropout = kwargs.get('dropout', 0.)

        if batch_norm:
            bn_kwargs = {key: kwargs[key] for key in ['eps', 'momentum', 'affine', 'track_running_stats'] if key in kwargs}
            batch_norm = getattr(torch.nn, f"BatchNorm{max(spatial_dim, 1)}d")(in_channels, **bn_kwargs)
        else:
            batch_norm = None

        return FNNLayer(linear, batch_norm=batch_norm, activation=activation, dropout=dropout)
    

class FNN(Sequential):
    """ Feedforward neural network.
    """
    def __init__(self, *layers: FNNLayer) -> None:
        super().__init__(*layers)

    @classmethod
    def from_config(cls, channels: Sequence[int], spatial_dim: int = 0, **kwargs) -> 'FNN':
        """ Create feedforward neural network from channel sizes, spatial dimension, and config kwargs.

        Parameters
        ----------
        channels
            Sequence of channel sizes
        spatial_dim, optional
            Spatial dimension, by default 0

        Returns
        -------
        FNN
            Feedforward neural network
        """
        kwargs = kwargs.copy()

        hidden_activation = kwargs.get('hidden_activation', 'GELU')
        output_activation = kwargs.get('output_activation', None)
        activations = ([hidden_activation] * (len(channels) - 2) + [output_activation])

        layers = []
        for i in range(len(channels) - 1):
            kwargs['activation'] = activations[i]
            layers.append(FNNLayer.from_config(channels[i], channels[i+1], spatial_dim, **kwargs))

        return FNN(*layers)
