from torch import Tensor


def butterfly(d1: Tensor, d2: Tensor, iv: Tensor, div_dz: Tensor, div_dzz: Tensor) -> Tensor:
    return (1 + div_dz * d1) * (1 + div_dz * d2) + iv * div_dzz


def calendar(r: Tensor, x: Tensor, iv: Tensor, div_dr: Tensor, div_dz: Tensor) -> Tensor:
    return ((iv - x * div_dz) / r + div_dr) / 2

