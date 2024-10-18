# Operator Deep Smoothing for Implied Volatility

This repository contains the code for the paper *Operator Deep Smoothing for Implied Volatility* available on [arXiv](http://arxiv.org/abs/2406.11520).
It contains the import package `op_ds`, which contains code components for *operator deep smoothing*, a general, discretization-invariant data interpolation method based on graph neural operators[^1].
Code components used to produce the results of the paper in the context of implied volatility smoothing reside in the folder `volatility_smoothing`.


## Installation

Simply pip-install to your environment (in editable mode) from the repository root:
```shell
(venv) ➜ pip install -e .
```

To run the notebooks, additionally install your favorite Jupyter version.

[^1]: Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar A., “Neural Operator: Learning Maps Between Function Spaces”, JMLR, 2021. doi:10.48550/arXiv.2108.08481.