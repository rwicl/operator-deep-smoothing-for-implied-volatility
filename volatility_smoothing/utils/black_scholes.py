import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Normal


normal = Normal(0, 1)


def normalizing_transforms(r, z, iv):
    a = - z / iv
    total_volatility = iv * r
    d1, d2 = a + total_volatility / 2, a - total_volatility / 2
    return d1, d2


def vega(r: Tensor, z: Tensor, iv: Tensor) -> Tensor:
    """Compute Black-Scholes vega.

    Parameters
    ----------
    r
        Square root of time-to-expiry
    z
        Normalized log-moneyness
    iv
        Black-Scholes volatility

    Returns
    -------
        Black-Scholes Vega
    """
    d1, _ = normalizing_transforms(r, z, iv)
    return normal.log_prob(d1).exp() * r


def normalized_option_price(r: Tensor, z: Tensor, iv: Tensor) -> Tensor:
    """Compute normalized option price, i.e. extrinsic option value.

    Parameters
    ----------
    r
        Square root of time-to-expiry
    z
        Normalized log-moneyness
    iv
        Black-Scholes volatility

    Returns
    -------
        Normalized option price
    """
    option_type = torch.ones_like(z)
    option_type[z < 0] = -1
    d1, d2 = normalizing_transforms(r, z, iv)
    strike = torch.exp(r * z)
    return F.relu(option_type * (normal.cdf(option_type * d1) - strike * normal.cdf(option_type * d2)))