# HyperIT

_Hyperscanning Analyses using Information-Theoretic Measures!_

[![Documentation Status](https://readthedocs.org/projects/hyperit/badge/?version=latest)](https://hyperit.readthedocs.io/en/latest/?badge=latest) ![Unit Testing](https://github.com/<username>/<repository>/actions/workflows/python-app.yml/badge.svg)


The HyperIT Class is a framework that calculates **Mutual Information** (MI), **Transfer Entropy** (TE), and **Integrated Information Decomposition** ($\Phi\text{ID}$) for **both hyperscanning and intra-brain analyses**. 

Handling continuous time-series data (epoched or otherwise), HyperIT computes these information-theoretic measures at **different frequency resolutions** and **different spatial scales of organisation** (micro, meso, and macro) — **compatible with EEG, MEG, fMRI, and fNIRS data**. Offers parameter customisation and estimator selection (Histogram/Binning, KSG, Box Kernel, Gaussian, and Symbolic) via JIDT. Most estimators are equipped with statistical significance testing based on permutation/bootstrapping approaches, too. Visualisations of MI/TE matrices and information atoms/lattices also provided. 


In all, HyperIT is designed to allow researchers to analyse various complex systems at different scales of organisation deploying information-theoretic measures, particularly focusing on neural time-series data in the context of hyperscanning. 

Read the JOSS pre-print here: "[HyperIT: A Python toolbox for an information-theoretic social neuroscience](https://github.com/EdoardoChidichimo/HyperIT/blob/master/HyperIT/paper/JOSS%20Article%20%E2%80%94%20HyperIT-%20A%20Python%20toolbox%20for%20an%20information-theoretic%20social%20neuroscience.pdf)"

## Usage

HyperIT uses a Class/OOP framework, allowing multiple instances of HyperIT objects (instantiated with different data). MI, TE, and $\Phi\text{ID}$ atoms can be computed by calling the following functions:

```python
from it import HyperIT


it = HyperIT(data1, data2, channel_names)

# ROIs can be specified, too 
it.roi(roi_list)

mi = it.compute_mi(estimator_type, calc_sigstats, vis)
te_xy, te_yx = it.compute_te(estimator_type, calc_sigstats, vis)
atoms = it.compute_atoms(tau, redundancy, vis)
```
For specific estimator types and general functionality, see Documentation and Tutorial.

## Dependencies
```
numpy
matplotlib
jpype
phyid
```
See: 

https://jpype.readthedocs.io/en/latest/ 

https://github.com/Imperial-MIND-lab/integrated-info-decomp/tree/main


## Acknowledgements
For MI and TE calculations, HyperIT depends on **JIDT** by Lizier and colleagues, accessible [here](https://github.com/jlizier/jidt), and published here: 

- Lizier, J. T. (2014). "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems", _Frontiers in Robotics and AI 1_(11). doi:[10.3389/frobt.2014.00011](http://dx.doi.org/10.3389/frobt.2014.00011) (pre-print: arXiv:[1408.3270](http://arxiv.org/abs/1408.3270))

For $\Phi\text{ID}$ atom calculations, HyperIT depends on **phyid** by the Imperial Mind Lab, with thanks to Pedro Mediano and Eric Ceballos Dominguez for providing the code.

- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D., Seth, A. K., & Barrett, A. B. (2021). Towards an extended taxonomy of information dynamics via Integrated Information Decomposition. https://doi.org/10.48550/ARXIV.2109.13186
- Luppi, A. I., Mediano, P. A. M., Rosas, F. E., Holland, N., Fryer, T. D., O’Brien, J. T., Rowe, J. B., Menon, D. K., Bor, D., & Stamatakis, E. A. (2022). A synergistic core for human brain evolution and cognition. Nature Neuroscience, 25(6), 771–782. https://doi.org/10.1038/s41593-022-01070-0
