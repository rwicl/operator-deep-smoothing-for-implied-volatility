from datetime import datetime
from random import uniform

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from volatility_smoothing.utils import black_scholes


class OptionsDataset(Dataset):

    def __init__(self, files: list, col_mapping: dict[str, str], r_lim: tuple[float, float] = (0.01, 1.), z_lim: tuple[float, float] = (-1.5, .5), subsample=False):
        """Create dataset for GNO training

        Mapping should be dictionary mapping all of the following to the position in the first axis of the loaded data tensors:
        
        * `option_type` (1 for Call option and -1 for Put option)
        * `time_to_maturity`
        * `log_moneyness`
        * `implied_volatility`
        * `bid`
        * `ask`
        * `discount_factor`
        * `underlying_forward`

        Parameters
        ----------
        files
            List of file paths, pickled Pytorch tensors (*.pt)
        col_mapping
            dict mapping column names to indices in data tensors
        r_lim
            Limits for (sqrt) time-to-expiry
        z_lim
            Limits for (normalized) log-moneyness
        subsample, optional
            Subsample input on each access, by default False
        """

        self.files = files
        self.mapping = col_mapping
        self.r_lim = r_lim
        self.z_lim = z_lim
        self.subsample = subsample

    def __len__(self):
        return len(self.files)
    
    def load_data(self, i: int) -> Data:
        
        raw_data = torch.load(self.files[i])
        date_string = self.files[i].split('CBOE_')[1][:-3]
        quote_datetime = datetime.strptime(date_string, '%Y-%m-%d-%H-%M-%S')

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

    def __getitem__(self, i: int) -> Data:
        return self.load_data(i)
