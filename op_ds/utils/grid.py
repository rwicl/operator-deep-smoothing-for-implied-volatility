from typing import Optional, Literal, Mapping

import torch
from torch import Tensor
from torch.nn  import Module
import matplotlib.pyplot as plt


class RectilinearGrid(Module, Mapping):
    """Rectilinear grid with arbitrary number of named axes.
    """

    def __init__(self, **axes: Tensor) -> None:
        super().__init__()

        axes = {k: v.sort()[0] for k, v in axes.items()}
        self._ax_idx = {k: i for i, k in enumerate(axes.keys())}
        self._ax_labels = {i: k for i, k in enumerate(axes.keys())}
        self.register_buffer('_meshgrid', torch.stack(torch.meshgrid(*axes.values(), indexing='ij'), dim=0))
    
    def __getitem__(self, key) -> Tensor:
        if isinstance(key, str):
            return self._meshgrid[self._ax_idx[key]]
        else:
            return self._meshgrid[key]
        
    def __iter__(self):
        return iter(self._ax_idx)
    
    def __len__(self):
        return len(self._ax_idx)
    
    def size(self):
        return self._meshgrid.size()[1:]
    
    def dim(self):
        return self._meshgrid.dim() - 1

    def extra_repr(self) -> str:
        return f"size={tuple(self.size())}"

    def flatten(self, layout: Literal['channel_first', 'channel_last']) -> Tensor:
        flattened = self._meshgrid.flatten(start_dim=1)
        if layout == 'channel_first':
            return flattened
        elif layout == 'channel_last':
            return flattened.transpose(1, 0)
        else:
            raise ValueError(f"Unknown layout: {layout}")
    
    def plot_surface(self, surface: Tensor, x: str | int = 0, y: str | int = 1, title: Optional[str] = None, ax: Optional[plt.Axes] = None, **kwargs) -> list[plt.Artist]:
        
        surface = surface.squeeze()
        
        if isinstance(x, str):
            x = self._ax_idx[x]
        if isinstance(y, str):
            y = self._ax_idx[y]

        artists = []
        if ax is None:
            ax = plt.subplot(projection='3d')
        artists.append(ax.plot_surface(self[x].cpu(), self[y].cpu(), surface.cpu(), **kwargs))    
        ax.set_xlabel(f'${self._ax_labels[x]}$')
        ax.set_ylabel(f'${self._ax_labels[y]}$')
        if title is not None:
            ax.set_title(title)
        return artists
    
    def differentiate(self, surface: Tensor, axis: str, order: int = 1) -> list[Tensor]:
        dim = self._ax_idx[axis]
        surface_dim = surface.dim() - self.dim() + self._ax_idx[axis]

        h_fw = self[axis].roll(-1, dims=dim) - self[axis]
        h_bw = self[axis] - self[axis].roll(1, dims=dim)

        left_idx = [slice(None)] * len(self)
        left_idx[dim] = 0
        right_idx = [slice(None)] * len(self)
        right_idx[dim] = -1

        diffs = [surface]
        for i in range(order):
            surface = diffs[-1]

            diff_forward = (surface.roll(-1, dims=surface_dim) - surface) / h_fw
            diff_backward = (surface - surface.roll(1, dims=surface_dim)) / h_bw
            diff = diff_forward / 2 + diff_backward / 2

            diff[..., *left_idx] = diff_forward[..., *left_idx]
            diff[..., *right_idx] = diff_backward[..., *right_idx]

            diffs.append(diff)

        return diffs[1:]
