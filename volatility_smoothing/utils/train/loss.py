from typing import Optional
from random import uniform
from pathlib import Path
from dataclasses import dataclass, field
from itertools import zip_longest, chain
from math import ceil, floor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch.distributions.normal import Normal
from tensordict import TensorDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from op_ds.utils.grid import RectilinearGrid
from op_ds.gno.gno import GNO
from volatility_smoothing.utils import arbitrage, errors, black_scholes
from volatility_smoothing.utils.train.dataset import GNOOptionsDataset
from volatility_smoothing.utils.train.edge_index import generate_edge_index
from volatility_smoothing.utils.svi import SVI
from volatility_smoothing.utils.slice_data import slice_data


normal = Normal(0., 1.)


@dataclass
class Loss:
    
    lim_r: tuple[float, float] = (0.01, 1.0)
    lim_z: tuple[float, float] = (-1.5, 0.5)
    step_r: Optional[float] = None
    step_z: Optional[float] = None    
    error_weights: dict[str, float] = field(default_factory=lambda: {'fit': 1., 'but': 10., 'cal': 10., 'reg_z': 0.01, 'reg_r': 0.01})
    eps_but: float = 1e-3
    eps_cal: float = 1e-3
    subsample_size: int = 35
    radius: float = 0.3

    def load_input(self, data, subsample_size: Optional[int] = None, radius: Optional[float] = None, 
                   step_r: Optional[float] = None, step_z: Optional[float] = None):

        if subsample_size is None:
            subsample_size = self.subsample_size

        if radius is None:
            radius = self.radius

        if step_r is None:
            step_r = self.step_r

        if step_z is None:
            step_z = self.step_z
    
        pos_x = torch.cat((data['r'], data['z']), dim=1)
        
        grids = ModuleList()

        # Generate grid for Butterfly arbitrage:
        r_axis = torch.arange(*self.lim_r, uniform(0.075, 0.125) if step_r is None else step_r)
        z_axis = torch.arange(*self.lim_z, 0.01 if step_z is None else step_z)
        grid = RectilinearGrid(r=r_axis, z=z_axis).to(pos_x.device)
        grids.append(grid)
    
        # Grids for Calendar arbitrage
        r_axis = torch.arange(*self.lim_r, 0.02 if step_r is None else step_r)
        z_axis = torch.arange(*self.lim_z, uniform(0.075, 0.125) if step_z is None else step_z)
        for i in range(r_axis.size(0) - 1):
            grid = RectilinearGrid(r=r_axis[i:i+2], z=z_axis).to(pos_x.device)
            grid._meshgrid[1, 1] = grid._meshgrid[1, 1] * r_axis[i] / r_axis[i + 1] # TODO Avoid this hack by generalizing grid class
            grids.append(grid)

        # Assemble GNO input and auxiliary data
        pos_y = torch.cat([grid.flatten('channel_last') for grid in grids], dim=0)  
        edge_index = generate_edge_index(pos_x, pos_y, subsample_size=subsample_size, radius=radius)

        input = TensorDict(**{
            'x': data['implied_volatility'],
            'pos_x': pos_x,
            'pos_y': pos_y,
            'edge_index': edge_index
        })

        aux = {
            'grids': grids
        }

        return input, aux

    @classmethod
    def read_output(cls, output, aux): 
        iv_x, iv_y = output
        grids = aux['grids']
        sections = [np.prod(grid.size()) for grid in grids]
        return (iv_x, *(iv.view(grids[i].size()) for i, iv in enumerate(torch.split(iv_y, sections, dim=0))))

    @classmethod
    def replication_error(cls, iv_target: Tensor, iv_predict: Tensor) -> Tensor:   
        error = (iv_predict - iv_target) / iv_target
        return error

    @classmethod
    def butterfly_term(cls, grid: RectilinearGrid, iv_surface: Tensor, include_second_derivatives: bool = False) -> Tensor:
        iv_surface = iv_surface.view(grid.size())
        div_dz, div_dzz = grid.differentiate(iv_surface, 'z', order=2)
        d1, d2 = black_scholes.normalizing_transforms(**grid, iv=iv_surface.clamp(min=0.005))
        but = arbitrage.butterfly(d1, d2, iv_surface, div_dz, div_dzz)        
        _, div_drr = grid.differentiate(iv_surface, 'r', order=2)
        
        if not include_second_derivatives:
            return but
        else:
            return but, div_dzz, div_drr

    @classmethod
    def calendar_term(cls, grid: RectilinearGrid, iv_grid: Tensor) -> Tensor:
        iv_grid = iv_grid.view(grid.size())
        return iv_grid[1:] / iv_grid[:-1].clamp(min=0.001) - grid['r'][:-1] / grid['r'][1:]

    @classmethod
    def errors(cls, data, output, aux):

        iv_predict, iv_but, *ivs_cal = cls.read_output(output, aux)
        grid_but, *grids_cal = aux['grids']

        replication_error = cls.replication_error(data['implied_volatility'], iv_predict)
        butterfly_error, div_dzz, div_drr = cls.butterfly_term(grid_but, iv_but, include_second_derivatives=True)
        calendar_error = torch.cat([cls.calendar_term(grid_cal, iv_cal) for grid_cal, iv_cal in zip(grids_cal, ivs_cal)], dim=0)

        return replication_error, butterfly_error, calendar_error, div_dzz, div_drr

    def loss(self, data, output, aux, dev=True):
        replication_error, butterfly_error, calendar_error, div_dzz, div_drr = self.errors(data, output, aux)
        
        losses = {
            'fit': (data['weight'] * replication_error.square()).mean().sqrt(),
            'but': F.relu(-butterfly_error - self.eps_but).mean(),
            'cal': F.relu(-calendar_error - self.eps_cal).mean(),
            'reg_z': div_dzz.square().mean().sqrt(),
            'reg_r': div_drr.square().mean().sqrt(),
        }
        
        l = sum([weight * losses[key] for key, weight in self.error_weights.items()])
            
        if not dev:
            return l
        else:
            with torch.no_grad():
                mape = replication_error.abs().mean()
                weighted_mape = (data['weight'] * replication_error.abs()).mean()
                        
            return l, {'loss': l, 'mape': mape, 'wmape': weighted_mape} | losses

    def compute_batch_loss(self, model: Module, batch: list[tuple[Data, TensorDict, TensorDict]], callback: callable, device: torch.device = None):
        
        batch_loss = 0
        loss_infos = []
        
        batch_size = len(batch)
        for data, input, aux in batch:
            data = data.to(device)
            input = input.to(device)
            aux['grids'] = [grid.to(device) for grid in aux['grids'] if grid is not None]

            output = model(**input)
            sample_loss, sample_loss_info = self.loss(data, output, aux)                
            sample_loss = sample_loss / batch_size
            callback(sample_loss)

            batch_loss = batch_loss + sample_loss
            loss_infos.append(sample_loss_info)

        return batch_loss, loss_infos

    def evaluate(self, model: Module, dataset: GNOOptionsDataset, device: torch.device = None, return_data=False, **kwargs):
        
        kwargs = kwargs.copy()
        storedir = kwargs.pop('storedir', None)
        logger = kwargs.pop('logger', None)

        data_storedir = None
        if storedir is not None:
            data_storedir = f"{storedir}/data"
            Path(data_storedir).mkdir(exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=self.collate_fn, shuffle=True, pin_memory=False, **kwargs)

        model = model.eval()

        rows_val = []
        rows_rel = []
        rows_fit = []

        data_list = []
        for data, input, aux in dataloader:

            data = data.to(device)
            input = input.to(device)
            aux['grids'] = [grid.to(device) for grid in aux['grids'] if grid is not None]
                            
            with torch.no_grad():
                output = model(**input)
            
            l, losses = self.loss(data, output, aux)
            rows_val.append({'quote_datetime': data.quote_datetime} | {'loss': l.item()} | {key: loss.item() for key, loss in losses.items()})

            iv_predict, iv_surface, *_ = self.read_output(output, aux)
            data_dict = {key: val.to('cpu', dtype=torch.float64).numpy() for key, val in data.items() if torch.is_tensor(val)}
            relative_error = errors.relative_error(iv_predict.to('cpu', dtype=torch.float64).numpy(), data_dict)
            spread_error = errors.spread_error(iv_predict.to('cpu', dtype=torch.float64).numpy(), data_dict)
            rows_rel.append({'quote_datetime': data.quote_datetime} | errors.descriptive_statistics(relative_error))
            rows_fit.append({'quote_datetime': data.quote_datetime} | errors.descriptive_statistics(spread_error))

            grid: RectilinearGrid = aux['grids'][0]
            div_dz, div_dzz = grid.differentiate(iv_surface, 'z', order=2)
            d1, d2 = black_scholes.normalizing_transforms(**grid, iv=iv_surface.clamp(min=0.001))
            g = arbitrage.butterfly(d1, d2, iv_surface, div_dz, div_dzz)
            
            data.iv_predict = iv_predict
            data.iv_surface = iv_surface
            data.normalized_spread =(data['ask'] - data['bid']) / (data['underlying_forward'] * data['discount_factor'])
            data.implied_density = normal.log_prob(-d2).exp() * g / (iv_surface * grid['r'])
            data.replication_error, data.butterfly_error, data.calendar_error, data.div_dzz, data.div_drr = self.errors(data, output, aux)
            data.grid = grid
            data_list.append(data)

            if data_storedir is not None:
                filepath = f"{data_storedir}/data_{data.quote_datetime.strftime('%Y-%m-%d-%H-%M-%S')}.pt"
                torch.save(data.cpu(), filepath)
            
            if logger is not None:
                logger.info(f"Evaluated quote datetime {data.quote_datetime}")
        
        df_val = pd.DataFrame(rows_val).set_index('quote_datetime').sort_index()
        df_rel = pd.DataFrame(rows_rel).set_index('quote_datetime').sort_index()
        df_fit = pd.DataFrame(rows_fit).set_index('quote_datetime').sort_index()

        if storedir is not None:
            start, end = df_val.index[0].strftime('%Y-%m-%d'), df_val.index[-1].strftime('%Y-%m-%d')
            df_val.to_csv(f"{storedir}/val_{start}-{end}.csv")
            df_rel.to_csv(f"{storedir}/rel_{start}-{end}.csv")
            df_fit.to_csv(f"{storedir}/fit_{start}-{end}.csv")

        if not return_data:
            return df_val, df_rel, df_fit
        else:
            return (df_val, df_rel, df_fit), data_list

    def collate_fn(self, data_list):
        data = data_list[0]
        return (data, *self.load_input(data))
    
    @staticmethod
    def format_loss_str(loss_infos):
        batch_size = len(loss_infos)
        loss_details = {k: [info[k] for info in loss_infos] for k in loss_infos[0]}
        loss_str = [
            f"loss: {sum(loss_details['loss']) / batch_size : .8f}",
            f"(mape: {sum(loss_details['mape']) / batch_size :> 10.3g}",
            f"wmape: {sum(loss_details['wmape']) / batch_size :> 10.3g}",
            f"fit pen: {sum(loss_details['fit']) / batch_size :> 10.3g}",
            f"cal pen: {sum(loss_details['cal']) / batch_size :> 10.3g}",
            f"but pen: {sum(loss_details['but']) / batch_size :> 10.3g}",
            f"reg_z pen: {sum(loss_details['reg_z']) / batch_size :> 10.3g}",
            f"reg_r pen: {sum(loss_details['reg_r']) / batch_size :> 10.3g})"
        ]
        return ', '.join(loss_str)

    def plot_example(self, model: GNO, data: Data, step: int = 3, **kwargs):

        figsize = kwargs.get('figsize', (9, 14))

        grid = RectilinearGrid(r=data.r.unique(), z=torch.arange(-1.5, .5, 0.01))
        pos_y = grid.flatten('channel_last')
        pos_x = torch.cat((data['r'], data['z']), dim=1)
        edge_index = generate_edge_index(pos_x, pos_y, subsample_size=self.subsample_size, radius=self.radius)

        input = {
            'x': data['implied_volatility'],
            'pos_x': pos_x,
            'pos_y': pos_y,
            'edge_index': edge_index
        }

        aux = {
            'grids': [grid],
        }

        with torch.no_grad():
            output = model(**input)
        iv_predict, iv_gno = self.read_output(output, aux)    
    
        expiries, slices = slice_data(data['r'], data['z'], data['implied_volatility'], iv_predict, data['vega'])
        ncols = floor(len(expiries) ** .5)
        nrows = ceil(len(expiries) / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        for i, ax in zip_longest(range(len(expiries)), chain(*axs)):
            if i is None:
                fig.delaxes(ax)
            else:
                r, z, iv_target, iv_predict, weight= slices[i]
                svi = SVI().fit({'r': r.numpy(), 'z': z.numpy(), 'implied_volatility': iv_target.numpy(), 'weight': weight})
                iv_svi = SVI.implied_volatility(z, *svi)
            
                z_plot = np.arange(-1.5, .5, 0.01)
                iv_svi = SVI.implied_volatility(z_plot, *svi)

                ax.scatter(z[::step], iv_target[::step], c='b', alpha=.5, s=8, marker='+', label='Mkt')
                ax.plot(z_plot, iv_svi, c='orange', alpha=.5, label='SVI')
                ax.plot(z_plot, iv_gno[i], c='g', alpha=.5, label='OpDS')
                ax.set_title(rf"$\tau={r[0]**2:.3f}$")
                ax.set_xlabel(r"$z = k / \sqrt{\tau}$")
                ax.legend()
                ax.grid()
                ax.set_aspect('auto')

        for col in range(ncols):
            last_ax = None
            for row in range(nrows):
                # check if matplotlib ax has been deleted:
                if not repr(axs[row, col]) == '<Axes: >':
                    last_ax = axs[row, col]
                else:
                    last_ax.xaxis.set_tick_params(labelbottom=True)
                    break

        return fig, axs