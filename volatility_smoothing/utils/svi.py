from dataclasses import dataclass, astuple
import numpy as np
from scipy.optimize import minimize

from volatility_smoothing.utils import arbitrage, black_scholes


@dataclass
class SVI:
    """Implementation of SVI (in its "raw" formulation; see https://arxiv.org/pdf/1204.0646)
    """

    a: float = 0.05
    b: float = 0.1
    rho: float = -.5
    sigma: float = 0.1
    m: float = 0.1

    def __array__(self):
        return np.array(astuple(self))
    
    def __iter__(self):
        return iter(astuple(self))

    @classmethod
    def implied_variance(cls, z, a: float, b: float, rho: float, sigma: float, m: float, nu=0) -> float:
        root_term = (z - m) ** 2 + sigma ** 2
        if nu == 0:
            return a + b * (rho * (z - m) + np.sqrt(root_term))
        elif nu == 1:
            return b * ((z - m) / np.sqrt(root_term) + rho)
        elif nu == 2:
            return b * sigma ** 2 / root_term ** 1.5
        else:
            raise NotImplementedError(f"Derivative order {nu} not implemented")

    @classmethod
    def implied_volatility(cls, z, *params, nu: int = 0) -> float:
        if nu == 0:
            w = cls.implied_variance(z, nu=0, *params)
            return w ** .5
        if nu == 1:
            iv = cls.implied_volatility(z, nu=0, *params)
            dw_dz = cls.implied_variance(z, nu=1, *params)
            return dw_dz / (2 * iv)
        if nu == 2:
            iv = cls.implied_volatility(z, nu=0, *params)
            dw_dz = cls.implied_variance(z, nu=1, *params)
            dw_dzz = cls.implied_variance(z, nu=2, *params)
            return dw_dzz / (2 * iv) - (dw_dz ** 2) / (4 * iv ** 3)
        else:
            raise NotImplementedError(f"Derivative order {nu} not implemented")

    def fit(self, data, **kwargs):
        kwargs = kwargs.copy()
        opt_kwargs = self.create_optimization_objective(data)
        kwargs.update(opt_kwargs)
        res = minimize(**kwargs)

        self.a, self.b, self.rho, self.sigma, self.m = res.x

        return self

    def create_optimization_objective(self, data) -> dict:
        r = np.unique(data['r']).item()
        z = data['z']
        iv_target = data['implied_volatility']

        try:
            weight = data['weight']
        except KeyError:
            weight = 1.

        def fun(x):
            iv_predict = SVI.implied_volatility(z, *x)
            error = (iv_target - iv_predict)
            loss = np.sqrt((weight * np.square(error)).mean())
            return loss
        
        def constraint_fun(x):
            iv = self.implied_volatility(z, *x)
            div_dz = self.implied_volatility(z, *x, nu=1)
            div_dzz = self.implied_volatility(z, *x, nu=2)
            d1, d2 = black_scholes.normalizing_transforms(r, z, iv)
            but = arbitrage.butterfly(d1, d2, iv, div_dz, div_dzz)
            g = np.concatenate((iv, but)) - 1e-4
            return g
        
        constraints = {'type': 'ineq', 'fun': constraint_fun}

        x0 = np.array(self)
        bounds = list({
            'a': (None, None), #((iv_target * r) ** 2).max()),
            'b': (0, 1),
            'rho': (-1, 1),
            'm': (-1.5, 0.5),
            'sigma': (1e-8, 2)
        }.values())

        return {
            'fun': fun, 
            'x0': x0,
            'bounds': bounds,
            'constraints': constraints
        }
