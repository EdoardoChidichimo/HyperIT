# HyperIT

_Hyperscanning Analyses using Information Theoretic Measures!_

[![Documentation](https://img.shields.io/badge/docs-passing-blue
)](https://github.com/EdoardoChidichimo/HyperIT/blob/main/docs/_build/html/index.html)

HyperIT is equipped to compute pairwise, multivariate **Mutual Information** (MI), **Transfer Entropy** (TE), and **Integrated Information Decomposition** (ΦID) for continuous time-series data. Compatible for both intra-brain and inter-brain analyses and for both epoched and unepoched data. Multiple estimator choices and parameter customisations (via JIDT) are available, including KSG, Kernel, Gaussian, Symbolic, and Histogram/Binning. Integrated statistical significance testing using permutation/boostrapping approach for most estimators. Visualisations of MI/TE matrices and information atoms/lattices also provided.

In all, HyperIT is designed to allow researchers to analyse various complex systems deploying information-theoretic measures, particularly focusing on neural time-series data in the context of hyperscanning. 

## Usage

HyperIT uses a Class/OOP framework, allowing you to make multiple instances HyperIT objects (each with different data). MI, TE, and ΦID atoms can be computed by calling the following functions:

```python
from it import HyperIT


it = HyperIT(data1, data2, channels)

mi = it.compute_mi(estimator_type, calc_sigstats, vis)
te_xy, te_yx = it.compute_te(estimator_type, calc_sigstats, vis)
atoms = it.compute_atoms(tau, redundancy, vis, plot_channels)
```
For specific estimator types and general functionality, see Documentation.

## Dependencies
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

## Acknowledgements
For MI and TE calculations, HyperIT depends on **JIDT** by Lizier and colleagues, accessible [here](https://github.com/jlizier/jidt), and published here: 

- Lizier, J. T. (2014). "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems", _Frontiers in Robotics and AI 1_(11). doi:[10.3389/frobt.2014.00011](http://dx.doi.org/10.3389/frobt.2014.00011) (pre-print: arXiv:[1408.3270](http://arxiv.org/abs/1408.3270))

For ΦID atom calculations, HyperIT depends on **phyid** by the Imperial Mind Lab, with thanks to Pedro Mediano and Eric Ceballos Dominguez for providing the code.

- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D., Seth, A. K., & Barrett, A. B. (2021). Towards an extended taxonomy of information dynamics via Integrated Information Decomposition. https://doi.org/10.48550/ARXIV.2109.13186
- Luppi, A. I., Mediano, P. A. M., Rosas, F. E., Holland, N., Fryer, T. D., O’Brien, J. T., Rowe, J. B., Menon, D. K., Bor, D., & Stamatakis, E. A. (2022). A synergistic core for human brain evolution and cognition. Nature Neuroscience, 25(6), 771–782. https://doi.org/10.1038/s41593-022-01070-0
