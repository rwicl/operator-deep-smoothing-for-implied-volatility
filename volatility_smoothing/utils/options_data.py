from abc import ABC, abstractmethod
from typing import Optional, Literal
import os
import re
from datetime import datetime
import warnings
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import py_vollib_vectorized

import torch
from torch.utils.data import Dataset


log = logging.getLogger(__name__)

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
    """Base class for options datasets
    
    Implement this class to add custom datasets.
    To do so, implement the abstract :meth:`~volformer.data.base.DatasetBase.download`-method.
    Then, upon initialization, the dataset will automatically download the data, process it (computing time-to-expiry, mid, forward, discount factor, implied volatility, ...), and cache it.
    The data is cached in the directory specified by the `OPDS_CACHE_DIR` environment variable (in a subdirectory named after the dataset class), and reprocessed only if the cache is not found or if the `force_reprocess` init-flag is set to `True`.
    This means that the dataset shouldn't really have a configurable initialization, since the reprocessing must be triggered manually.
    Required per-user information should be pulled from environment variables.
    """
    

    def __init__(self, force_reprocess: bool = False) -> None:
        """Initialize dataset

        Makes sure cache of processed files exists (just checks existence of respective directory).
        If it doesn't, creates it by invoking :meth:`load_data`, processing the data, and then writing one file per surface to the cache.
        Once initialized, surfaces are loaded on the fly from the cache, one-by-one per access.

        Parameters
        ----------
        force_reprocess : bool, optional
            If True, reprocess data even if cache exists. By default `False`.
        """

        if os.getenv("OPDS_CACHE_DIR") is None:
            raise ValueError("OPDS_CACHE_DIR environment variable not set")
        cache_dir = Path(os.getenv("OPDS_CACHE_DIR")) / self.__class__.__name__

        if not cache_dir.exists() or force_reprocess:
            cache_dir.mkdir(parents=True, exist_ok=True)
            df = (self
                    .load_data()
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
            self._cache_data(df, cache_dir)
        
        log.info("Assembling and sorting index of cached files.")
        self.file_paths = sorted(cache_dir.glob(f"{self.__class__.__name__}_*.pt"))
        self.quote_datetimes = pd.DatetimeIndex([self._get_quote_datetime(file) for file in self.file_paths], name='quote_datetime')
        log.info(f"Created index of {len(self.file_paths)} files.")

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
        return torch.load(self.file_paths[i])
        
    def _get_quote_datetime(self, filepath):
        date_string = str(filepath).split(f'{type(self).__name__}_')[1].split('.')[0]
        quote_datetime = datetime.strptime(date_string, '%Y-%m-%d-%H-%M-%S')
        return quote_datetime

    def _cache_data(self, df: pd.DataFrame, cache_dir: Optional[str] = None) -> None:
        quote_datetimes = df.index.get_level_values('quote_datetime').unique()
        for i, quote_datetime in enumerate(quote_datetimes):
            data = df.xs(quote_datetime, level='quote_datetime')
            datestr = quote_datetime.strftime("%Y-%m-%d-%H-%M-%S")
            filepath = os.path.join(cache_dir, f"{type(self).__name__}_{datestr}.pt")
            torch.save(data, filepath)

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


class SPXOptionsDataset(OptionsDataset):
    """S&P 500 options dataset."""
    
    @classmethod
    def load_data(cls) -> pd.DataFrame:
        
        data_path = Path(os.environ['OPDS_CBOE_SPX_DATA_DIR'])
        zip_files = [file for file in os.listdir(data_path) if file.endswith('.zip')]

        usecols = ['root','expiration', 'quote_datetime', 'option_type', 'strike', 'bid', 'ask']

        def read_zip_files(zip_files):
            for i, f in enumerate(zip_files, start=1):
                log.info(f"Reading file {i} of {len(zip_files)}")
                temp_df = pd.read_csv(data_path / f, usecols=usecols)
                temp_df = temp_df.query('root == "SPX"')
                yield temp_df

        df = pd.concat(read_zip_files(zip_files), ignore_index=True)
        log.info("Finished concatenating all files")
        
        df = df.rename(columns={'expiration': 'expiry_datetime'})
        
        return df



if __name__ == "__main__":
    os.environ['OPDS_CACHE_DIR'] = "/Users/ruben/.cache/opds"
    os.environ['OPDS_CBOE_SPX_DATA_DIR'] = "/Users/ruben/data/cboe/subsample"
    dataset = SPXOptionsDataset(force_reprocess=True)
