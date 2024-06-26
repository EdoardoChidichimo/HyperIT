# HyperIT

_Hyperscanning Analyses using Information-Theoretic Measures!_

[![Documentation Status](https://readthedocs.org/projects/hyperit/badge/?version=latest)](https://hyperit.readthedocs.io/en/latest/?badge=latest) ![Unit Testing](https://github.com/EdoardoChidichimo/HyperIT/actions/workflows/unit-tests.yml/badge.svg?random=12345) 


The HyperIT Class is a framework that calculates **Mutual Information** (MI), **Transfer Entropy** (TE), and **Integrated Information Decomposition** ($\Phi\text{ID}$) for **both hyperscanning and intra-brain analyses**. 

Handling continuous time-series data (epoched or otherwise), HyperIT computes these information-theoretic measures at **different spatial scales of organisation** (micro, meso, and macro) — **compatible with EEG, MEG, fMRI, and fNIRS data**. Offers parameter customisation and estimator selection (Histogram/Binning, KSG, Box Kernel, Gaussian, and Symbolic) via JIDT. Most estimators are equipped with statistical significance testing based on permutation/bootstrapping approaches, too. Visualisations of MI/TE matrices are also provided. 

In all, HyperIT is designed to allow researchers to analyse various complex systems at different scales of organisation deploying information-theoretic measures, particularly focusing on neural time-series data in the context of hyperscanning. 

Read the JOSS pre-print here: "[HyperIT: A Python toolbox for an information-theoretic social neuroscience](https://github.com/EdoardoChidichimo/HyperIT/blob/master/HyperIT/paper/JOSS%20Article%20%E2%80%94%20HyperIT-%20A%20Python%20toolbox%20for%20an%20information-theoretic%20social%20neuroscience.pdf)"

## Usage

HyperIT uses a Class/OOP framework, allowing multiple instances of HyperIT objects (instantiated with different data). MI, TE, and $\Phi\text{ID}$ atoms can be computed by calling the following functions:

```python
from hyperit import HyperIT
from phyid.utils import PhiID_atoms_abbr

# Only needs to be called once
HyperIT.setup_JVM()

# Gather your data here ...

# Create instance
it = HyperIT(data1, data2, channel_names, verbose)

# ROIs can be specified and then reset back to default
it.roi(roi_list)
it.reset_roi()

# Calculate Mutual Information and Transfer Entropy
mi = it.compute_mi(estimator='kernel',
                   include_intra=True,
                   epoch_average=False
                   calc_sigstats=True,
                   vis=True,
                   plot_epochs=[1,6], # use -1 to plot all epochs
                   kwargs) # Pass estimator-specific parameters here

te = it.compute_te(estimator='gaussian',
                   include_intra=False,
                   epoch_average=True
                   calc_sigstats=True,
                   vis=True,
                   kwargs) 

# Calculate Integrated Information Decomposition
atoms = it.compute_atoms(tau=5, redundancy='MMI', include_intra=True, epoch_average=True)
print({key: value for key, value in zip(PhiID_atoms_abbr, atoms)})
```

For specific estimator types and general functionality, see Documentation and Tutorial.


## Installation

To install HyperIT, simply use pip:

```bash
pip install git+https://github.com/EdoardoChidichimo/HyperIT.git
```


## Dependencies
```
numpy
scipy
matplotlib
jpype1
phyid
```
See: 

https://jpype.readthedocs.io/en/latest/ 

https://github.com/Imperial-MIND-lab/integrated-info-decomp/tree/main


## Acknowledgements
For MI and TE calculations, HyperIT depends on **JIDT** by Lizier and colleagues, accessible [here](https://github.com/jlizier/jidt), and published here: 

- Lizier, J. T. (2014). "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems", _Frontiers in Robotics and AI 1_(11). doi:[10.3389/frobt.2014.00011](http://dx.doi.org/10.3389/frobt.2014.00011) (pre-print: arXiv:[1408.3270](http://arxiv.org/abs/1408.3270))

For $\Phi\text{ID}$ atom calculations, HyperIT depends on **phyid** from Imperial Mind Lab, with thanks to Pedro Mediano and Eric Ceballos Dominguez for providing the code and guidance.

- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D., Seth, A. K., & Barrett, A. B. (2021). Towards an extended taxonomy of information dynamics via Integrated Information Decomposition. https://doi.org/10.48550/ARXIV.2109.13186
- Luppi, A. I., Mediano, P. A. M., Rosas, F. E., Holland, N., Fryer, T. D., O’Brien, J. T., Rowe, J. B., Menon, D. K., Bor, D., & Stamatakis, E. A. (2022). A synergistic core for human brain evolution and cognition. Nature Neuroscience, 25(6), 771–782. https://doi.org/10.1038/s41593-022-01070-0
