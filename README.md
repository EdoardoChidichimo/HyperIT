# HyperIT

Hyperscanning Analyses using Information Theoretic Measures!

## USAGE

```python
from it import HyperIT

it = HyperIT(data1, data2, channels)

mi = it.compute_mi(estimator_type, calc_sigstats, vis)
te_xy, te_yx = it.compute_te(estimator_type, calc_sigstats, vis)
atoms = it.compute_atoms(tau, redundancy, vis, plot_channels)
