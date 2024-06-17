from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch_geometric.nn import MessagePassing

from op_ds.utils.fnn import FNN


class GNOLayer(MessagePassing):
    """Graph Neural Operator (GNO) layer.
    """
    def __init__(self, channels: int, transform: Module,
                 local_linear: bool = False, local_bias: bool = True,
                 activation: Optional[Module] = None, 
                 lifting: Optional[Module] = None, projection: Optional[Module] = None) -> None:
        super().__init__(aggr='mean', flow='source_to_target')

        self.transform = transform

        if local_linear:
            self.local = FNN.from_config((channels, channels), batch_norm=False, bias=False)
        else:
            self.register_parameter('local', None)

        if local_bias:
            self.bias = torch.nn.Parameter(Tensor(channels))
        else:
            self.register_parameter('bias', None)
        
        if activation is not None:
            self.add_module('activation', activation)
        else:
            self.register_parameter('activation', None)

        if lifting is not None:
            self.add_module('lifting', lifting)
        else:
            self.register_parameter('lifting', None)

        if projection is not None:
            self.add_module('projection', projection)
        else:
            self.register_parameter('projection', None)

        self.channels = channels

        self.reset_parameters()
        
    def __repr__(self) -> str:
        return super(MessagePassing, self).__repr__()
    
    def extra_repr(self) -> str:
        return 'channels={}, local_linear={}, local_bias={}'.format(
            self.channels, self.local is not None, self.bias is not None
        )

    def update(self, aggr_out: Tensor, pos: Tensor, v: Tensor, x: Tensor) -> Tensor:
        if self.local is not None:
            aggr_out = aggr_out + self.local(v)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def message(self, pos_i, pos_j, v_i, v_j, x_i, x_j) -> Tensor:
        return self.transform(pos_i=pos_i, pos_j=pos_j, v_i=v_i, v_j=v_j, x_i=x_i, x_j=x_j)

    def forward(self, edge_index: Tensor, pos: Tensor, v: Tensor, x: Tensor) -> Tensor:
        if self.lifting is not None:
            v = self.lifting(v)
        w = self.propagate(edge_index, pos=pos, v=v, x=x)
        if self.projection is not None:
            w = self.projection(w)
        if self.activation is not None:
            w = self.activation(w)
        return w
    

class GNO(Module):
    """Graph Neural Operator (GNO) model.

    Given by a sequence of GNO layers.
    """
    def __init__(self, *gno_layers: GNOLayer, in_channels: int = 1) -> None:
        super().__init__()
        self.gno_layers = ModuleList(gno_layers)
        self.in_channels = in_channels

    def forward(self, x: Tensor, pos_x: Tensor, pos_y: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """GNO forward propagation.

        Parameters
        ----------
        x
            Input data, shape (n, in_channels)
        pos_x
            Coordinate locations of input data x, shape (n, domain_dim)
        pos_y
            Coordinate locations at which to compute the output y, shape (m, domain_dim)
        edge_index
            Graph connectivity, shape (2, e)

        Returns
        -------
        tuple[Tensor, Tensor]
            Output data at pos_x, shape (n, out_channels)
            Output data at pos_y, shape (m, out_channels)
        """

        n = pos_x.size(0)
        m = pos_y.size(0)
        x = torch.cat((x, torch.full((m, self.in_channels), fill_value=0., dtype=x.dtype, device=x.device)))
        pos = torch.cat((pos_x, pos_y))
        
        w = x
        for gno_layer in self.gno_layers:
            w = gno_layer(edge_index, pos=pos, v=w, x=x)
        
        return torch.split(w, (n, m), dim=0)
        
    def __str__(self) -> str:
        extra_lines = []
        child_lines = []

        for key, module in self._modules.items():
            child_lines.append(f"({key}): {str(module)}")
        lines = extra_lines + child_lines
        
        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str 