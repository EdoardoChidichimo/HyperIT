---
title: 'HyperIT: A Python toolbox for an information-theoretic social neuroscience'
tags:
  - Python
  - Social Neuroscience
  - Hyperscanning
  - Information Theory
authors:
  - name: Edoardo Chidichimo
    orcid: 0000-0002-1179-9757
    affiliation: "1" 
    corresponding: true
  - name: Richard A. I. Bethlehem
    orcid: 0000-0002-0714-0685
    affiliation: "1,2"
  - name: Guillaume Dumas
    orcid: 0000-0002-2253-1844
    affiliation: "3,4"
affiliations:
  - name: Department of Psychology, University of Cambridge, Cambridge, UK
    index: 1
  - name: Brain Mapping Unit, Department of Psychiatry, University of Cambridge, Cambridge, UK
    index: 2
  - name: CHU Sainte-Justine Research Centre, Department of Psychiatry, Université de Montréal, Montréal, QC, Canada 
    index: 3
  - name: Mila–Quebec AI institute, Montréal, QC, Canada
    index: 4

date: 21 April 2024
bibliography: paper.bib
header-includes:
  - |
    \DeclareMathOperator*{\SumInt}{%
      \mathchoice%
      {\ooalign{$\displaystyle\sum$\cr\hidewidth$\displaystyle\int$\hidewidth\cr}}%
      {\ooalign{\raisebox{.14\height}{\scalebox{.7}{$\textstyle\sum$}}\cr\hidewidth$\textstyle\int$\hidewidth\cr}}%
      {\ooalign{\raisebox{.2\height}{\scalebox{.6}{$\scriptstyle\sum$}}\cr$\scriptstyle\int$\cr}}%
      {\ooalign{\raisebox{.2\height}{\scalebox{.6}{$\scriptscriptstyle\sum$}}\cr$\scriptscriptstyle\int$\cr}}%
    }
---


# Summary

`HyperIT` is a toolbox for neuroscientists who are interested in uncovering the informational dynamics of socially-engaged brains and their complex neural dynamics. Using information theory and its various measures, `HyperIT` offers neuroscientists an intuitive and standardised toolbox to analyse their neuroimaging or neurophysiological recordings, making the toolbox versatile to multiple modalities (including EEG, MEG, fNIRS, and fMRI), whilst simultaneously handling epochality, specified frequency resolutions, and spatial scales of organisation. 


# Statement of need

For the past two decades, social neuroscience has addressed the interpersonal neural dynamics of interacting persons through the hyperscanning paradigm [@dumas_inter-brain_2010]. Synchrony and other statistical properties have been prioritised in these analyses, although most measures remain of the parametric type despite neural activity and coordination dynamics often being distinctly nonlinear. `HyperIT` presents the first principled and unified analysis framework for hyperscanning data using nonparametric measures. More than this, for the first time it combines and offers canonical measures of dependency and causality with the most recent measures of information dynamics. 

Between interacting persons, we observe coordination and communication on various scales of organisation whose unit of transfer we can loosely describe as \textit{information}. Whilst the idea of interpersonal information transfer may appear intuitive \textit{prima facie}, it is not exactly clear how to define information, let alone measure its content, distribution, transfer, storage, modification, and other informational dynamics. Thus, an elegant definition and quantification of information that is \textit{mathematically precise and consistent} is essential, particularly one that is \textit{content-invariant} and \textit{model-free}; i.e., making no assumptions on the information content itself nor the statistical distributions or relationship model between random variables.

Information theory, lauded as \textit{the} mathematical theory of communication [@shannon_mathematical_1948], lends itself well to this cause, proffering domain-generality and information as its common currency. Originally birthed for the development of communications engineering, various information-theoretic measures have found vast applicability in computer science (e.g., Kolmogorov complexity), economics (e.g., portfolio theory and Kelly gambling), statistics (e.g., Fisher information and hypothesis testing), and probability theory (e.g., limit theorems and large derivations; @cover_elements_2006). Information theory has recently earned a notable place in neuroscience, too [@timme_tutorial_2018].  

Of relevance for neuroscientific endeavours, information theory provides measures that can detect linear \textit{and} nonlinear dependencies between continuous time-series signals, allowing researchers to analyse both the correlation and causation between two or more random variables. In all, whilst standard measures of correlation and prediction are only sensitive to linear dependencies and only describe the variable's overall relationship, information-theoretic analyses and measures can quantify and more comprehensively describe the dynamics of complex systems that may demonstrate nonlinear dependencies whilst maintaining a particular robustness to noise. `HyperIT` computes well-recognised and validated information-theoretic measures that, for the first time, can be simultaneously applied to both hyperscanning and intra-brain analyses for various neural recordings whilst uniquely including the most recent measures, decomposing these datasets into their constituent informational dynamics. Usefully, and unlike other libraries, `HyperIT` is equipped to handle epoched and event-based data as well as specifying both frequency bands and the level of organisation by comparing channels pairwise or by regions of interest. These measures include \textit{(I) mutual information}, \textit{(II) transfer entropy}, and \textit{(III) integrated information decomposition}, described below. 


# Measures

\textit{Mutual information} ($I(X;Y)$) is a positively-biased, symmetric measure indicating how much (Shannon) information is shared between two random variables $X$ and $Y$ (\autoref{eq:mi}). One may also interpret MI as quantifying the “distance” between the joint probability distribution ($p_{XY}$) and the product of marginal probability distributions ($p_X \otimes p_Y$), such that $I(X;Y)=0$ \textit{iff} $p_{XY} = p_{X} \otimes p_{Y}$. Thus, it is understood as a special instance of the Kullback-Leibler divergence measure.

\begin{equation}\label{eq:mi}
    I(X;Y) = \SumInt_{\mathcal{X}}\SumInt_{\mathcal{Y}}p_{XY}(x,y)\log_2\frac{p_{XY}(x,y)}{p_X(x)p_Y(y)}. 
\end{equation}

\textit{Transfer entropy} ($TE_{Y \rightarrow X}$) is a measure that non-parametrically measures the statistical connectivity and time-directed transfer of information between random variables or processes. It is often taken as the non-linear equivalent of Granger causality and, indeed, equivalence has been demonstrated under Gaussianity [@barnett_granger_2009]. Specifically, this measure uses conditional mutual information to quantify the reduction in uncertainty about the future of one process given knowledge about another variable and its own history (\autoref{eq:te}).

\begin{align}\label{eq:te} \notag
    TE_{Y\rightarrow X}^{(k,l,u,\tau)} &= I(\mathbf{Y}_{t-u}^{(l,\tau_Y)}; X_t | \mathbf{X}_{t-1}^{(k,\tau_X)}) \\ 
    &= \SumInt_{\substack{x_t, \mathbf{x}_{t-1}^{(k,\tau_X)} \in \mathcal{X}, \\ \mathbf{y}_{t-u}^{(l,\tau_Y)} \in \mathcal{Y}}} p\left(x_t, \mathbf{x}_{t-1}^{(k,\tau_X)}, \mathbf{y}_{t-u}^{(l,\tau_Y)}\right) \log_2 \left( \frac{p\left(\mathbf{y}_{t-u}^{(l,\tau_Y)}, x_t | \mathbf{x}_{t-1}^{(k,\tau_X)}\right)}{p\left(x_t | \mathbf{x}_{t-1}^{(k,\tau_X)}\right)} \right),
\end{align} with parameters including embedding history length for source ($l$) and target ($k$), embedding delay for source ($\tau_Y$) and target ($\tau_X$), and some causal delay or interaction lag $u$.

More recently, approaches to exhaustively decompose a multivariate system’s informational structure has described three modes; namely, information about a target variable may be redundant (Rdn): information that is shared between variables; unique (Unq): information that is specific to a single variable; or synergistic (Syn): information that is only learnt from the conjunction of multiple sources and not individually, with the exclusive-OR function being a canonical example [@williams_nonnegative_2010]. These so-called “partial information atoms” are non-overlapping and form an additive set, exhaustively describing the informational composition of a multivariate system (\autoref{eq:pid}). 

\begin{align}\label{eq:pid}
    I(X_1, X_2; Y) = &\text{Syn}(X_1, X_2; Y) + \\ \notag
    &\text{Unq}(X_1; Y) + \text{Unq}(X_2; Y) + \\ \notag
    &\text{Rdn}(X_1, X_2; Y). \notag
\end{align} 

A recent development, termed \textit{integrated information decomposition}, extends this decomposition to multi-source and multi-target continuous time-series random variables to decompose information dynamics into various qualitative modes including information storage, copy, transfer, erasure, downward causation, causal decoupling, and upward causation [@mediano_beyond_2019; @mediano_towards_2021]. This measure specifically decomposes the time-delayed mutual information ($t' > t$) between two multivariate processes (\autoref{eq:phi-id}).

\begin{align}\label{eq:phi-id}
    I(\mathbf{X}_t;\mathbf{X}_{t'}) = &\text{Syn}(X_t^1,X_t^2; \mathbf{X}_{t'}) + \\ \notag
    &\text{Unq}(X_t^1; \mathbf{X}_{t'}|X_t^2) + \text{Unq}(X_t^2; \mathbf{X}_{t'}|X_t^1) + \\ \notag
    &\text{Rdn}(X_t^1,X_t^2; \mathbf{X}_{t'}). \\ \notag
\end{align}

# Functionality

`HyperIT`, then, addresses these approaches and offers a user-friendly, class-based toolbox where users can create a `HyperIT` object passing two multivariate sets of continuous time-series signals (dimensions including epochality and multiple channels). As mentioned, users can choose to bandpass filter their signals at specified frequency bands as well as specify the scale of organisation by setting the "regions of interest" property; namely, specifying whether (and which) channels are computed pairwise or as clusters with one another for all measures, making micro-, meso-, and macro-scale analysis readily available. From here, users can call mutual information, transfer entropy, and integrated information decomposition functions specifying estimation type and estimation-specific parameters (outlined in documentation). Estimators include (a) histogram/binning, (b) box kernel, (c) Gaussian, (d) k-nearest neighbour, and (e) symbolic approaches. Users may also choose to conduct statistical significance testing via permutation testing for each calculation and optionally visualise the measure matrices. Importantly, the toolbox relies upon the Java Information Dynamics Toolkit for computing mutual information and transfer entropy with various estimators [@lizier_jidt_2014], although `HyperIT` makes this Java-oriented toolbox accessible and compatible for Python coders who may in turn enjoy a diverse ecosystem of other scientific libraries, most notably MNE and HyPyP [@ayrolles_hypyp_2021]. Moreover, `HyperIT` offers its own histogram and symbolic mutual information estimators that are unavailable in JIDT. Users will need to download and store the `infodynamics.jar` file locally as well as ensuring the `jpype` dependency is installed. Finally, `HyperIT` uniquely intgrates the `phyid` package to compute integrated information atoms [@luppi_synergistic_2022; @mediano_towards_2021] and offer users to compare between standard measures and the more precise informational decomposition of their data. Thus, `HyperIT` offers a distinct interoperability and compatibility with other Python libraries whilst providing unique features of handling hyperscanning data and information dynamics, and versatility with epochality, frequency resolutions, and spatial scales of organisation.

In all, social neuroscience researchers working with continuous time-series signals (including fMRI, EEG, MEG, and fNIRS), particularly in the context of hyperscanning, will be, for the first time, able to comfortably compute common and powerful information-theoretic measures using our `HyperIT` toolbox.



# Acknowledgements

Thanks to Richard Bethlehem, Guillaume Dumas, Pedro A. M. Mediano, Eric C. Dominguez, and Andrea I. Luppi for their support and contributions to the fruition of this project.

# References