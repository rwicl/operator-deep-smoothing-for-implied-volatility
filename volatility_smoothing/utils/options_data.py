from abc import ABC, abstractmethod
from typing import Optional, Literal
import os
import re
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import py_vollib_vectorized

import torch
from torch.utils.data import Dataset


def imply_borrow(x: pd.DataFrame, k: int = 3, atm_bound: Optional[float] = None) -> pd.Series:
    if atm_bound is None:
        calls = x[x[('mid', 'P')] > x[('mid', 'C')]].head(k)
        puts = x[x[('mid', 'P')] <= x[('mid', 'C')]].head(k)
        atm = pd.concat((calls, puts))
    else:
        moneyness = x.index.get_level_values('strike') / x['underlying_mid', 'C']
        atm = x.loc[(1 - atm_bound <= moneyness) & (moneyness <= 1 + atm_bound)]#
        
    y = atm[('mid', 'C')] - atm[('mid', 'P')]
    x = atm.index.get_level_values(level='strike').values
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    discount_factor = -m
    underlying_forward = c / discount_factor

    d = {}
    d['discount_factor'] = discount_factor
    d['underlying_forward'] = underlying_forward

    return pd.Series(d, index=['discount_factor', 'underlying_forward'])


class OptionsDataset(Dataset, ABC):
    """Options datasets base class
    """
    
    def __init__(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None, return_as: Literal['csv', 'pt'] = 'pt') -> None:
        """Create options dataset from either raw data or previously processed (cached) data

        If not providing ``data_dir`` during init, then must provide ``cache_dir`` (to populate a ``cache_dir`` use ).

        Parameters
        ----------
        data_dir, optional
            Directory from which to load the raw data, by default None
        cache_dir, optional
            Cache directory, where to cache or load processed data from, by default None
        return_as, optional
            Format to return data in, by default 'pt'
        """
        if data_dir is not None:
            self._data = (self
                          .load_data(data_dir)
                          .dropna(subset=['expiry_datetime', 'quote_datetime', 'strike', 'option_type', 'bid', 'ask'])
                          .astype({'quote_datetime': 'datetime64[ns]',
                                    'expiry_datetime': 'datetime64[ns]',
                                    'strike': float,
                                    'option_type': 'str',
                                    'bid': float,
                                    'ask': float})
                          .pipe(self.add_implieds)
                          .get(self.columns)
                          .sort_index())
            self.quote_datetimes = self._data.index.get_level_values('quote_datetime').unique()
        else:
            self._data = [os.path.join(cache_dir, filename) for filename in os.listdir(cache_dir) if filename.endswith(f".{return_as}")]
            self.quote_datetimes = pd.DatetimeIndex([self._get_quote_datetime(file) for file in self._data], name='quote_datetime')

        self.return_as = return_as

    @classmethod
    @abstractmethod
    def load_data(cls, data_dir: str) -> pd.DataFrame:
        """Load the data from the given directory and process

        Return a pandas dataframe with the following columns:

        * ``expiry_datetime``: expiry timestamp of the option
        * ``quote_datetime``: timestamp of the quote (recorded when)
        * ``option_type``: 'C' for call and 'P' for put
        * ``strike``: strike price of the option
        * ``bid``: bid price of the option
        * ``ask``: ask price of the option

        Parameters
        ----------
        data_dir
            The path to the directory containing the data

        Returns
        -------
            The processed data
        """
        pass

    def __len__(self):
        return len(self.quote_datetimes)
    
    def __getitem__(self, i: int):
        return self.get_surface(i, return_as=self.return_as)
        
    def _get_quote_datetime(self, filepath):
        date_string = filepath.split(f'{type(self).__name__}_')[1].split('.')[0]
        quote_datetime = datetime.strptime(date_string, '%Y-%m-%d-%H-%M-%S')
        return quote_datetime

    def get_surface(self, i: int, return_as: Literal['pt', 'csv']) -> pd.DataFrame:
        # Load surface
        if isinstance(self._data, list):
            if self.return_as == 'pt':
                surface = torch.load(self._data[i], weights_only=True)
            elif self.return_as == 'csv':
                surface = pd.read_csv(self._data[i])
        elif isinstance(self._data, pd.DataFrame):
            surface = self._data.xs(self.quote_datetimes[i], level='quote_datetime')
            if return_as == 'pt':
                # surface['option_type'] = surface['option_type'].map({'C': 1., 'P': -1.})
                surface = surface.reset_index(drop=True)
                surface = torch.tensor(surface.values.T, dtype=torch.float32)
        else:
            raise ValueError("Unknown data type")
        
        return surface

    def cache_data(self, cache_dir: Optional[str] = None) -> None:
        for i, quote_datetime in enumerate(self.quote_datetimes):
            data = self[i]
            datestr = quote_datetime.strftime("%Y-%m-%d-%H-%M-%S")
            filepath = os.path.join(cache_dir, f"{type(self).__name__}_{datestr}.{self.return_as}")
            if self.return_as == 'pt':
                torch.save(data, filepath)
            elif self.return_as == 'csv':
                data.to_csv(filepath)
            else:
                raise ValueError(f"Unknown return_as: {self.return_as}")
    
    @property
    def columns(self):
        return ['option_type', 'time_to_maturity', 'log_moneyness', 'implied_volatility', 'bid', 'ask', 'discount_factor', 'underlying_forward']

    @staticmethod
    def add_implieds(df):
        
        df = (df
            .assign(
                time_to_maturity=lambda df: ((df['expiry_datetime'] - df['quote_datetime']).dt.total_seconds()) / (365 * 24 * 60 * 60),  # 252 <- 252 counting seems to be wrong
                mid=lambda df: (df['bid'] + df['ask']) / 2,
            )
            .query('time_to_maturity > 0')  # drop time_to_maturity == 0
            .set_index(['quote_datetime', 'expiry_datetime', 'strike'])
            .pivot(columns='option_type')
            .get([
                'time_to_maturity',
                'mid',
                'bid',
                'ask'
            ])
        )
        # Throw away any quote which somewhere have underlying zero-price! Mus
        #idx = (df['underlying_bid'].groupby(level='quote_datetime').min() > 0).all(axis=1)  # Do not allow any underlying bid to be zero
        #idx =  & (data['ask'].groupby(level=['quote_datetime', 'expiration']).min() > 0).all(axis=1))
        #idx = ((description.loc[:, ('underlying_bid', slice(None), 'min')] > 0).all(axis=1))  # Do not allow any underlying bid to be zero
        #      # & (description.loc[:, ('ask', slice(None), 'max')] > 0).all(axis=1))  # Only take where there is a non-zero ask
        # df = df[idx.reindex(df.index, level='quote_datetime')].copy()

        borrow = df.groupby(['quote_datetime', 'expiry_datetime']).apply(imply_borrow)

        df['discount_factor', 'C'] = borrow['discount_factor']
        df['discount_factor', 'P'] = borrow['discount_factor']
        df['underlying_forward', 'C'] = borrow['underlying_forward']
        df['underlying_forward', 'P'] = borrow['underlying_forward']
        df['log_moneyness', 'C'] = np.log(df.index.get_level_values('strike') / df['underlying_forward', 'C'])
        df['log_moneyness', 'P'] = np.log(df.index.get_level_values('strike') / df['underlying_forward', 'P'])
        
        S = df['underlying_forward', 'C'].values
        K = df.index.get_level_values('strike').values
        t = df['time_to_maturity', 'C'].values
        r = np.zeros_like(t) #(- np.log(df['discount_factor']) / t).values
        flag = np.full(r.shape, fill_value='c')
        df['implied_volatility', 'C'] = py_vollib_vectorized.vectorized_implied_volatility(df['mid', 'C'].values, S, K, t, r, flag, return_as='array')

        S = df['underlying_forward', 'P'].values
        K = df.index.get_level_values('strike').values
        t = df['time_to_maturity', 'P'].values
        r = np.zeros_like(t) #(- np.log(df['discount_factor']) / t).values
        flag = np.full(r.shape, fill_value='p')
        df['implied_volatility', 'P'] = py_vollib_vectorized.vectorized_implied_volatility(df['mid', 'P'].values, S, K, t, r, flag, return_as='array')
        df = df.swaplevel(axis=1)

        calls = df['C'].assign(option_type=1.)
        puts = df['P'].assign(option_type=-1.)
        df = pd.concat((calls, puts))

        return df


class CBOEOptionsDataset(OptionsDataset):
    r"""Dataset for S&P 500 Index options data as provided by the CBOE DataShop"""

    @classmethod
    def load_data(cls, data_dir: str) -> pd.DataFrame:

        # get the path of the unique csv file in the directory:
        # warn in case where there are multiple files in the directory   

        csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No csv files found in {data_dir}")
        else:
            warnings.warn("Loading first file only")
            filepath = os.path.join(data_dir, csv_files[0])

        data = (pd.read_csv(filepath)
                  .query('root == "SPX"')
                  .rename(columns={'expiration': 'expiry_datetime'}))
        
        return data

