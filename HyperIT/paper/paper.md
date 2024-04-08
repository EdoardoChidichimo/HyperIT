---
title: 'HyperIT: A Python toolbox for an information-theoretic social neuroscience'
tags:
  - Python
  - social neuroscience
  - hyperscanning
  - Shannon information theory
  - mutual information
  - transfer entropy
  - integrated information decomposition
authors:
  - name: Edoardo Chidichimo
    orcid: 0000-0002-1179-9757
    affiliation: "1" 
affiliations:
 - name: Department of Psychology, University of Cambridge, UK
   index: 1
date: 8 April 2024
bibliography: paper.bib
---

# Summary

`HyperIT` is an open-source, class-based Python toolbox designed for an information-theoretic social neuroscience. Specifically designed for hyperscanning paradigms (simultaneous neuroimaging and neurophysiological recordings of social interactions), `HyperIT` is equipped to compute Shannon information theory measures including mutual information, transfer entropy, and integrated information decomposition for continuous time-series signals at different scales of organisation. The toolbox allows customisation of parameters for up to five different empirical estimators (including histogram, box kernel, kNN, Gaussian, and symbolic methods), statistical significance testing, and matrix visualisations. `HyperIT` integrates, depends upon, and provides an interface for the Java Information Dynamics Toolkit [@lizier_jidt_2014] and `phyid` library [@luppi_synergistic_2022; @mediano_towards_2021].


# Statement of need

For the past two decades, social neuroscience has addressed the interpersonal neural dynamics of interacting persons through the hyperscanning paradigm [@dumas_inter-brain_2010]. Synchrony and other statistical properties have been prioritised in these analyses, although most measures remain of the parametric type despite neural activity and coordination dynamics often being distinctly nonlinear. 

Between interacting persons, we observe coordination and communication on various scales of organisation whose unit of transfer we can loosely describe as \textit{information}. Whilst the idea of interpersonal information transfer may appear intuitive prima facie, it is not exactly clear how to define information, let alone measure its content, distribution, transfer, storage, modification, and other informational dynamics. Thus, a precise definition and quantification of information is essential, particularly one that is both \textit{content-invariant} and \textit{model-free} — making no assumption on the information content itself — and \textit{mathematically precise and consistent}.

Information theory, lauded as \textit{the} mathematical theory of communication [@shannon_mathematical_1948], lends itself well to this cause, proffering domain-generality and information as its common currency. By definition, then, information theory is model-free, making no assumptions on statistical distributions or relationship model between variables. In any case, originally birthed for the development of communications engineering, various information-theoretic measures have found vast applicability in computer science (e.g., Kolmogorov complexity), economics (e.g., portfolio theory and Kelly gambling), statistics (e.g., Fisher information and hypothesis testing), and probability theory (e.g., limit theorems and large derivations; Cover and Thomas, 2006), information theory has earned a notable place in neuroscience [@timme_tutorial_2018].  

Of relevance for neuroscientific endeavours, information theory provides measures that can detect linear and nonlinear dependencies between continuous time-series signals, allowing researchers to analyse both the correlation and causation between two or more variables. In all, whilst standard measures of correlation and prediction are only sensitive to linear dependencies and only describe the variable's overall relationship, information-theoretic analyses and measures can quantify and more comprehensively describe the dynamics of complex systems that may demonstrate nonlinear dependencies whilst maintaining a particular robustness to noise. The information-theoretic measures available in `HyperIT` include \textit{mutual information}, \textit{transfer entropy}, and \textit{integrated information decomposition}, described below. 


# Measures

Mutual information ($I(X;Y)$) is a positively-biased, symmetric measure indicating how much (Shannon) information is shared between two random variables $X$ and $Y$ (Equation \autoref{eq:mi}). One may also interpret MI as quantifying the “distance” between the joint probability distribution ($p_{XY}$) and the product of marginal probability distributions ($p_X \otimes P_Y$), such that $I(X;Y)=0$ \textit{iff} $P_{XY} = P_{X} \otimes P_{Y}$. This is also understood as a special instance of the Kullback-Leibler divergence measure.

\DeclareMathOperator*{\SumInt}{%
\mathchoice%
  {\ooalign{$\displaystyle\sum$\cr\hidewidth$\displaystyle\int$\hidewidth\cr}}
  {\ooalign{\raisebox{.14\height}{\scalebox{.7}{$\textstyle\sum$}}\cr\hidewidth$\textstyle\int$\hidewidth\cr}}
  {\ooalign{\raisebox{.2\height}{\scalebox{.6}{$\scriptstyle\sum$}}\cr$\scriptstyle\int$\cr}}
  {\ooalign{\raisebox{.2\height}{\scalebox{.6}{$\scriptstyle\sum$}}\cr$\scriptstyle\int$\cr}}
}

\begin{equation} \label{eq:mi}
    I(X;Y) = \SumInt_{\mathcal{X}}\SumInt_{\mathcal{Y}}p_{XY}(x,y)\log_2\frac{p_{XY}(x,y)}{p_X(x)p_Y(y)} 
\end{equation}

Transfer entropy ($TE_{Y \rightarrow X}$) is a measure that non-parametrically measures the statistical coherence and time-directed transfer of information between random variables or processes. It is often taken as the non-linear equivalent of Granger causality and, indeed, they are shown to be equivalent under Gaussianity [@barnett_granger_2009]. Specifically, this measure uses conditional mutual information to quantify the reduction in uncertainty about the future of one process given knowledge about another variable and its own history (Equation \autoref{eq:te}).

\begin{align} \label{eq:te} \notag
    TE_{Y\rightarrow X}^{(k,l,u,\tau)} &= I(\bm{Y}_{t-u}^{(l,\tau_Y)}; X_t | \bm{X}_{t-1}^{(k,\tau_X)}) \\ 
    &= \SumInt_{\substack{x_t, \bm{x}_{t-1}^{(k,\tau_X)} \in \mathcal{X}, \\ \bm{y}_{t-u}^{(l,\tau_Y)} \in \mathcal{Y}}} p\left(x_t, \bm{x}_{t-1}^{(k,\tau_X)}, \bm{y}_{t-u}^{(l,\tau_Y)}\right) \log_2 \left( \frac{p\left(\bm{y}_{t-u}^{(l,\tau_Y)}, x_t | \bm{x}_{t-1}^{(k,\tau_X)}\right)}{p\left(x_t | \bm{x}_{t-1}^{(k,\tau_X)}\right)} \right)
\end{align} with parameters including embedding history length for source ($l$) and target ($k$), embedding delay for source ($\tau_Y$) and target ($\tau_X$), and some causal delay or interaction lag $u$.

More recently, approaches to exhaustively decompose a multivariate system’s informational structure has described three modes; namely, information about a target variable may be redundant (Rdn): information that is shared between variables; unique (Unq): information that is specific to a single variable; or synergistic (Syn): information that is only learnt from the conjunction of multiple sources and not individually, with the exclusive-OR function being a canonical example [@williams_nonnegative_2010]. These so-called “partial information atoms” are non-overlapping and form an additive set, exhaustively describing the informational composition of a multivariate system (Identity \autoref{eq:pid}). A recent development, termed integrated information decomposition, extends this decomposition to multi-source and multi-target continuous time-series variables to decompose information dynamics into various qualitative modes including information storage, copy, transfer, erasure, downward causation, causal decoupling, and upward causation [@mediano_beyond_2019; @mediano_towards_2021]. This measure specifically decomposes the time-delayed mutual information between two multivariate processes (Equations \autoref{eq:phi-id}) 

\begin{equation} \label{eq:pid}
    I(Y; X_1, X_2) = \text{Syn}(Y; X_1, X_2) + \text{Unq}(Y; X_1) + \text{Unq}(Y; X_2) + \text{Rdn}(Y; X_1, X_2) 
\end{equation} 

\begin{align} \label{eq:phi-id}
    I(\bm{X}_t;\bm{X}_{t'}) &= \text{Syn}(X_t^1,X_t^2; \bm{X}_{t'}) + \text{Unq}(X_t^1; \bm{X}_{t'}|X_t^2) + \text{Unq}(X_t^2; \bm{X}_{t'}|X_t^1) + \text{Rdn}(X_t^1,X_t^2; \bm{X}_{t'}) \\ \notag
\end{align}

# Functionality

`HyperIT`, then, addresses these approaches and offers a user-friendly, class-based toolbox where users can create a `HyperIT` object passing two multivariate sets of continuous time-series signals (dimensions including epochality and multiple channels). Users can choose to specify the scale of organisation, too, by setting the "regions of interest" property; namely, specifying whether (and which) channels are computed pairwise or as clusters with one another for all measures, making micro-, meso-, and macro-scale analysis readily available. From here, users can call mutual information and transfer entropy functions specifying estimation type and estimation-specific parameters (outlined in documentation). Estimators include (a) histogram/binning, (b) box kernel, (c) Gaussian, (d) k-nearest neighbour, and (e) symbolic approaches. Users can also select to conduct statistical significance testing via permutation testing for each calculation and choose to visualise the measure matrices. Importantly, the toolbox relies heavily upon the Java Information Dynamics Toolkit for computing mutual information and transfer entropy with various estimators [@lizier_jidt_2014], although `HyperIT` makes this Java-oriented toolbox accessible and compatible for Python coders. Users will need to store the `infodynamics.jar` file locally in their working directory as well as ensuring the `jpype` dependency is installed. Finally, `HyperIT` calls the `phyid` package to compute integrated information atoms [@luppi_synergistic_2022; @mediano_towards_2021]. Notably, we encourage users to also deploy analyses available from the Hyperscanning Python Pipeline (HyPyP) of which `HyperIT` is compatible with [@ayrolles_hypyp_2021].

In all, social neuroscience researchers working with continuous time-series signals (including fMRI, EEG, MEG, and fNIRS), particularly in the context of hyperscanning, should be able to comfortably compute common and powerful information-theoretic measures with the `HyperIT` toolbox.



# Acknowledgements

Thanks to Pedro A. M. Mediano, Eric C. Dominguez, Andrea I. Luppi, Richard Bethlehem, and Guillaume Dumas during the genesis of this project.

# References