# HyperIT

Hyperscanning Analyses using Information Theoretic Measures!

## USAGE

HyperIT uses a Class/OOP framework, allowing you to make multiple instances HyperIT objects (each with different data). MI, TE, and atoms can be computed by calling the following functions:

```python
from it import HyperIT


it = HyperIT(data1, data2, channels)

mi = it.compute_mi(estimator_type, calc_sigstats, vis)
te_xy, te_yx = it.compute_te(estimator_type, calc_sigstats, vis)
atoms = it.compute_atoms(tau, redundancy, vis, plot_channels)
```

## DEPENDENCIES
```
numpy
matplotlib
PIL
jpype
phyid
```
See: 
https://jpype.readthedocs.io/en/latest/ 
https://github.com/Imperial-MIND-lab/integrated-info-decomp/tree/main

## ACKNOWLEDGEMENTS
This code depends on JIDT by Lizier and colleagues, accessible [here](https://github.com/jlizier/jidt), and published here: 

Lizier, J. T. (2014). "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems", _Frontiers in Robotics and AI 1_(11). doi:[10.3389/frobt.2014.00011](http://dx.doi.org/10.3389/frobt.2014.00011) (pre-print: arXiv:[1408.3270](http://arxiv.org/abs/1408.3270))


