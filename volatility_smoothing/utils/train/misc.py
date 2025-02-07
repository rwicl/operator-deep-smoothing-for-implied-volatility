import torch

from op_ds.gno.gno import GNOLayer, GNO
from op_ds.gno.kernel import NonlinearKernelTransformWithSkip
from op_ds.utils.fnn import FNN


def create_gno():
    num_hidden_layers = 3
    in_channels = 1  # fixed for this problem
    out_channels = 1  # fixed for this problem
    gno_channels = 16
    channels = (in_channels, *(num_hidden_layers * (gno_channels,)), out_channels)  # in total four layers
    spatial_dim = 2  # fixed for this problem
    fnn_hidden_channels = 64

    gno_layers = []

    for i in range(m := (len(channels) - 1)):
        lifting = FNN.from_config((channels[i], fnn_hidden_channels, gno_channels), hidden_activation='gelu', batch_norm=False)
        projection = None if i < m - 1 else FNN.from_config((gno_channels, fnn_hidden_channels, channels[i+1]), hidden_activation='gelu', batch_norm=False)
        transform = NonlinearKernelTransformWithSkip(in_channels=gno_channels, out_channels=gno_channels, skip_channels=in_channels, spatial_dim=spatial_dim, hidden_channels=(fnn_hidden_channels, fnn_hidden_channels), hidden_activation='gelu', batch_norm=False)

        if i == 0:
            local_linear = False
        else:
            local_linear = True
        
        activation = torch.nn.GELU() if i < m - 1 else torch.nn.Softplus(beta=0.5)
        
        gno_layer = GNOLayer(gno_channels, transform=transform, local_linear=local_linear, local_bias=True,
                            activation=activation, lifting=lifting, projection=projection)
        gno_layers.append(gno_layer)

    gno = GNO(*gno_layers, in_channels=in_channels)
    return gno


def load_checkpoint(path: str, device=None):
    model = create_gno().to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def format_loss_str(loss_infos, batch_size):
    loss_details = {k: [info[k] for info in loss_infos] for k in loss_infos[0]}
    loss_str = [
        f"mape: {sum(loss_details['mape']) / batch_size :> 10.3g}",
        f"wmape: {sum(loss_details['wmape']) / batch_size :> 10.3g}",
        f"vol pen: {sum(loss_details['rel']) / batch_size :> 10.3g}",
        f"cal pen: {sum(loss_details['cal']) / batch_size :> 10.3g}",
        f"but pen: {sum(loss_details['but']) / batch_size :> 10.3g}",
        f"reg_z pen: {sum(loss_details['reg_z']) / batch_size :> 10.3g}",
        f"reg_r pen: {sum(loss_details['reg_r']) / batch_size :> 10.3g}"
    ]
    return loss_str