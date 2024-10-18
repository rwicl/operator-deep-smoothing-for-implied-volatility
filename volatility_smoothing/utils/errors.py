from typing import Mapping

import numpy as np

from py_vollib_vectorized import vectorized_black_scholes


def spread_error(iv_predict: np.ndarray, data: Mapping[str, np.ndarray]) -> dict[str, float]:
    K = np.exp(np.asarray(data['log_moneyness'])).squeeze()
    S = np.ones_like(K)
    r = np.zeros_like(K)
    t = np.asarray(data['time_to_maturity']).squeeze()
    flag = ['c' if K >= 1 else 'p' for K in K]

    iv = np.asarray(iv_predict).squeeze()
    df = np.asarray(data['discount_factor']).squeeze()
    fw = np.asarray(data['underlying_forward']).squeeze()
    mid_predict = df * fw * vectorized_black_scholes(flag, S, K, t, r, iv, return_as='array')
    mid = np.asarray(((data['bid'] + data['ask']) / 2)).squeeze()
    spread = np.asarray(((data['ask'] - data['bid']) / 2)).squeeze()

    spread_error = np.abs((mid_predict - mid)) / spread

    return spread_error


def relative_error(iv_predict: np.ndarray, data: Mapping[str, np.ndarray]) -> dict[str, float]:
    iv_target = data['implied_volatility']
    relative_error = np.abs((iv_predict - iv_target) / iv_target)

    return relative_error


def descriptive_statistics(error: np.ndarray) -> dict[str, float]:

    return {
        'mean': error.mean(),
        'std': error.std(),
        'min': error.min(),
        '%05': np.quantile(error, q=0.05),
        '%25':  np.quantile(error, q=0.25),
        '%50': np.quantile(error, q=0.5),
        '%75': np.quantile(error, q=0.75),
        '%95': np.quantile(error, q=0.95),
        'max': error.max(),
    }
