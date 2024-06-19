import torch
from torch import Tensor


def pairwise_differences(x: Tensor, y: Tensor) -> Tensor:
    return (x.view(x.size(0), 1, *x.size()[1:]) - y.view(1, y.size(0), *y.size()[1:]))


def generate_edge_index(pos_x: Tensor, pos_y: Tensor, subsample_size: int = 50, radius: float = 0.5, include_self_loops: bool = False) -> Tensor:
    pos = torch.cat((pos_x, pos_y), dim=0)
    distances_r = pairwise_differences(pos_x[..., 0], pos[..., 0]).abs()
    idx = torch.argsort(distances_r, dim=0, stable=True)
    distances_r = torch.gather(distances_r, 0, idx)

    idx_bound = torch.argmin((distances_r <= radius).to(dtype=torch.int), dim=0)

    edge_index_list = []
    for i in torch.arange(pos.size(0)):

        k = idx_bound[i]
        step = int(k // subsample_size) + 1
        s = idx[0:k:step, i]
        
        t = i.repeat(s.size())
        edge_index_list.append(torch.stack((s, t.to(s.device)), dim=1))
        if include_self_loops and not (i == s).any() and i < idx.size(0):
            edge_index_list.append(torch.tensor([[i, i]], device=s.device, dtype=s.dtype))

    edge_index = torch.cat(edge_index_list).transpose(0, 1).contiguous()

    return edge_index
