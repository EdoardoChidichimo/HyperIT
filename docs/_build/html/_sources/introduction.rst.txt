Introduction
============

The HyperIT Class is a framework that calculates **Mutual Information** (MI), **Transfer Entropy** (TE), and **Integrated Information Decomposition** ($\Phi\text{ID}$) for **both hyperscanning and intra-brain analyses**. 

Handling continuous time-series data (epoched or otherwise), HyperIT computes these information-theoretic measures at **different frequency resolutions** and **different spatial scales of organisation** (micro, meso, and macro) — **compatible with EEG, MEG, fMRI, and fNIRS data**. Offers parameter customisation and estimator selection (Histogram/Binning, KSG, Box Kernel, Gaussian, and Symbolic) via JIDT. Most estimators are equipped with statistical significance testing based on permutation/bootstrapping approaches, too. Visualisations of MI/TE matrices and information atoms/lattices also provided. 

In all, HyperIT is designed to allow researchers to analyse various complex systems at different scales of organisation deploying information-theoretic measures, particularly focusing on neural time-series data in the context of hyperscanning. 