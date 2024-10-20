from random import uniform

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from volatility_smoothing.utils import black_scholes
from volatility_smoothing.utils.options_data import OptionsDataset


class GNOOptionsDataset(Dataset):

    def __init__(self, options_dataset: OptionsDataset, r_lim: tuple[float, float] = (0.01, 1.), z_lim: tuple[float, float] = (-1.5, .5), subsample=False):
        """Create dataset for GNO training

        Parameters
        ----------
        options_dataset
            Options dataset
        col_mapping
            dict mapping column names to indices in data tensors
        r_lim
            Limits for (sqrt) time-to-expiry
        z_lim
            Limits for (normalized) log-moneyness
        subsample, optional
            Subsample input on each access, by default False
        """

        self.options_dataset = options_dataset
        self.r_lim = r_lim
        self.z_lim = z_lim
        self.subsample = subsample
        self.mapping = {
            'option_type': 0,
            'time_to_maturity': 1,
            'log_moneyness': 2,
            'implied_volatility': 3,
            'bid': 4,
            'ask': 5,
            'discount_factor': 6,
            'underlying_forward': 7
        }

    def __len__(self):
        return len(self.options_dataset)
    
    def __getitem__(self, i: int) -> Data:
        
        quote_datetime = self.options_dataset.quote_datetimes[i]
        raw_data = self.options_dataset[i]

        option_type_idx = self.mapping['option_type']
        log_moneyness_idx = self.mapping['log_moneyness']
        call_idx = (raw_data[option_type_idx] == 1) & (raw_data[log_moneyness_idx] > 0)
        put_idx = (raw_data[option_type_idx] == -1) & (raw_data[log_moneyness_idx] <= 0)
        nan_idx = ~raw_data.isnan().any(dim=0)

        raw_data = raw_data[list(self.mapping.values())][:, (call_idx | put_idx) & nan_idx]
        
        if self.subsample:
            subsample_idx = torch.rand(raw_data.size(1)) <= uniform(0.6, 1.2)
            raw_data = raw_data[:, subsample_idx]   
        
        keys = list(self.mapping.keys())

        t, k = raw_data[keys.index('time_to_maturity')], raw_data[keys.index('log_moneyness')]

        r = t.sqrt()
        z = k / r
        
        zero_ask_idx = raw_data[keys.index('ask')] > 0
        domain_idx = (self.r_lim[0] <= r) & (r <= self.r_lim[1]) & (self.z_lim[0] <= z) & (z <= self.z_lim[1])
        idx = zero_ask_idx & domain_idx
        
        r = r[idx, None]
        z = z[idx, None]

        data_dict = dict(zip(self.mapping.keys(), raw_data[:, idx, None]))

        vega = black_scholes.vega(r, z, data_dict['implied_volatility'])
        weight = torch.maximum(vega / vega.mean(), torch.tensor(1.0))

        return Data(r=r, z=z, **data_dict, vega=vega, weight=weight, num_nodes=r.size(0), quote_datetime=quote_datetime)
