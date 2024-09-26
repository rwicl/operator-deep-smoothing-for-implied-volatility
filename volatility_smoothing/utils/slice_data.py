import torch
from torch import Tensor


def slice_data(axis: Tensor, *features: Tensor):
    """Perform group-by operation on features based on axis

    Parameters
    ----------
    axis
        Axis to group by
    features
        Features to group

    Returns
    -------
    Tuple[Tensor, List[Tensor]]
        Unique values of axis and list of features grouped by axis
    """
    axis = axis.squeeze()
    features = torch.stack([axis] + [d.squeeze() for d in features])
    slices = []
    axis_uniques, inverse = axis.unique(return_inverse=True)
    for i, r in enumerate(axis_uniques):
        slices.append(features[..., inverse == i])

    return axis_uniques, slices