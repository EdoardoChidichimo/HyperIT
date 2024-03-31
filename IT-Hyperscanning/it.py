import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from PIL import Image, ImageDraw
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from jpype import *
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr






class HyperIT(ABC):
    """ HyperIT: A Comprehensive Framewowrk for Conducting Hyperscanning Information Theoretic (HyperIT) Analyses.

        HyperIT is equipped to compute multivariate Mutual Information (MI), Transfer Entropy (TE), and Integrated Information Decomposition (ΦID).
        Supports both intra-brain and inter-brain analyses, epoched and unepoched time-series signals.  
        Designed to be flexible and user-friendly, allowing for multiple estimator choices and parameter customisations (using JIDT).
        Provides statistical significance testing using permutation/boostrapping approach for most estimators.
        Allows visualisations of MI/TE matrices and information atoms/lattices.

    Args:
        ABC (_type_): Abstract Base Class for HyperIT.

    Note: This class requires numpy, matplotlib, PIL, jpype (with the infodynamics.jar file), and phyid as dependencies.
    """
    def __init__(self, data1: np.ndarray, data2: np.ndarray, channel_names: List[str], verbose: bool = False):
        """ Creates HyperIT object containing time-series data and channel names for analysis. 
            Automatic data checks for consistency and dimensionality, identifying whether analysis is to be intra- or inter-brain.

            Determines whether epochality of data.
                - If data is 3 dimensional, data is assumed to be epoched with shape    (epochs, channels, time_points).
                - If data is 2 dimensional, data is assumed to be unepoched with shape          (channels, time_points).

        Args:
            data1               (np.ndarray): Time-series data for participant 1.
            data2               (np.ndarray): Time-series data for participant 1.
            channel_names        (List[str]): A list of strings representing the channel names for each participant. [[channel_names_p1], [channel_names_p2]] or [[channel_names_p1]] for intra-brain.
            verbose         (bool, optional): Whether constructor and analyses should output details and progress. Defaults to False.
        """

        self._setup_JVM(os.getcwd()) # NEED TO CHANGE TO FILE LOCATION

        self.data1: np.ndarray = data1
        self.data2: np.ndarray = data2
        self.channel_names: List[str] = channel_names
        self.inter_brain: bool = not np.array_equal(data1, data2)
        self.is_epoched: bool = data1.ndim == 3
        self.initialise_parameter = None
        self.verbose = verbose

        self.__check_data()

        if self.verbose:
            print("HyperIT object created successfully.")
            if self.is_epoched:
                print(f"Epoched data detected. Assuming each signal has shape ({self.n_epo} epochs, {self.n_chan} channels, {self.n_samples} time_points). Ready to conduct {'Inter-Brain' if self.inter_brain else 'Intra-Brain'} analysis.")
            else:
                print(f"Unepoched data detected. Assuming each signal has shape ({self.n_chan} channels, {self.n_samples} time_points). Ready to conduct {'Inter-Brain' if self.inter_brain else 'Intra-Brain'} analysis.")


    def __del__(self):
        """ Destructor for HyperIT object. Ensures JVM is shutdown upon deletion of object. """
        try:
            shutdownJVM()
            if self.verbose:
                print("JVM has been shutdown.")
        except Exception as e:
            if self.verbose:
                print(f"Error shutting down JVM: {e}")

    def __repr__(self) -> str:
        """ String representation of HyperIT object. """
        analysis_type = 'Hyperscanning' if self.inter_brain else 'Intra-Brain'
        channel_info = f"{self.channel_names[0][0]}"  # Assuming self.channel_names[0] is a list of channel names for the first data set
        
        # Adding second channel name if inter_brain analysis is being conducted
        if self.inter_brain:
            channel_info += f" and {self.channel_names[0][1]}"

        return (f"HyperIT Object: \n"
                f"{analysis_type} Analysis with {self.n_epo} epochs, {self.n_chan} channels, "
                f"and {self.n_samples} time points. \n"
                f"Channel names passed: \n"
                f"{channel_info}.")

    def __len__(self) -> int:
        """ Returns the number of epochs in the HyperIT object. """
        return self.n_epo
    
    def __str__(self) -> str:
        """ String representation of HyperIT object. """
        return self.__repr__()
    

    @staticmethod        
    def _setup_JVM(working_directory: str) -> None:
        if(not isJVMStarted()):
            jarLocation = os.path.join(working_directory, "..", "..", "infodynamics.jar")
            # Usually, just specifying the current working directory (cwd) would suffice; if not, use specific location, e.g., below
            jarLocation = "/Users/edoardochidichimo/Desktop/MPhil_Code/IT-Hyperscanning/infodynamics.jar"

            if (not(os.path.isfile(jarLocation))):
                exit("infodynamics.jar not found (expected at " + os.path.abspath(jarLocation) + ") - are you running from demos/python?")

            startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


    def __check_data(self) -> None:
        """ Checks the consistency and dimensionality of the time-series data and channel names. Sets the number of epochs, channels, and time points as object variables.

        Ensures:
            - Data are numpy arrays.
            - Data shapes are consistent.
            - Data dimensions are either 2 or 3 dimensional.
            - Channel names are in correct format and match number of channels in data.
        """

        if not all(isinstance(data, np.ndarray) for data in [self.data1, self.data2]):
            raise ValueError("Time-series data must be numpy arrays.")
        
        if self.data1.shape != self.data2.shape:
            raise ValueError("Time-series data must have the same shape for both participants.")
    
        if self.data1.ndim not in [2,3]:
            raise ValueError(f"Unexpected number of dimensions in time-series data: {self.data1.ndim}. Expected 2 dimensions (channels, time_points) or 3 dimensions (epochs, channels, time_points).")

        if not isinstance(self.channel_names, (list, np.ndarray)) or isinstance(self.channel_names[0], str):
            raise ValueError("Channel names must be a list of strings or a list of lists of strings for inter-brain analysis.")
    
        if not self.inter_brain and isinstance(self.channel_names[0], list):
            self.channel_names = [self.channel_names] * 2

        if self.is_epoched:
            self.n_epo, self.n_chan, self.n_samples = self.data1.shape
        else:
            self.n_epo = 1
            self.n_chan, self.n_samples = self.data1.shape

        n_channels = self.data1.shape[1] if self.is_epoched else self.data1.shape[0]
        if any(len(names) != n_channels for names in self.channel_names[0]):
            raise ValueError("The number of channels in time-series data does not match the length of channel_names.")

    @staticmethod
    def __setup_JArray(a: np.ndarray) -> JArray:
        """ Converts a numpy array to a Java array for use in JIDT."""

        a = (a).astype(np.float64) 
        try:
            ja = JArray(JDouble, 1)(a)
        except Exception: 
            ja = JArray(JDouble, 1)(a.tolist())
        return ja


    def _mi_hist(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Calculates Mutual Information using Histogram/Binning Estimator for time-series signals."""

        @staticmethod
        def calc_fd_bins(X: np.ndarray, Y: np.ndarray) -> int:
            """Calculates the optimal frequency-distribution bin size for histogram estimator using Freedman-Diaconis Rule."""

            fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
            fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
            fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
            return fd_bins

        pairwise = np.zeros((self.n_epo, 1))

        for epo_i in range(self.n_epo):

            X, Y = (s1[epo_i, :], s2[epo_i, :]) if self.is_epoched else (s1, s2)

            j_hist, _, _ = np.histogram2d(X, Y, bins=calc_fd_bins(X, Y))
            pxy = j_hist / np.sum(j_hist)  # Joint probability distribution

            # Marginals
            px = np.sum(pxy, axis=1) 
            py = np.sum(pxy, axis=0) 

            # Entropies
            Hx = -np.sum(px * np.log2(px + np.finfo(float).eps))
            Hy = -np.sum(py * np.log2(py + np.finfo(float).eps))
            Hxy = -np.sum(pxy * np.log2(pxy + np.finfo(float).eps))

            result = Hx + Hy - Hxy

            pairwise[epo_i] = result

        return pairwise

    def _mi_symb(self, s1: np.ndarray, s2: np.ndarray, l: int = 1, m: int = 3) -> float:
        """Calculates Mutual Information using Symbolic Estimator for time-series signals."""

        symbol_weights = np.power(m, np.arange(m))
        pairwise = np.zeros((self.n_epo, 1))

        def symb_symbolise(X: np.ndarray, l: int, m: int) -> np.ndarray:
            Y = np.empty((m, len(X) - (m - 1) * l))
            for i in range(m):
                Y[i] = X[i * l:i * l + Y.shape[1]]
            return Y.T

        def symb_normalise_counts(d) -> None:
            total = sum(d.values())        
            return {key: value / total for key, value in d.items()}
        
        for epo_i in range(self.n_epo):

            X, Y = (s1[epo_i, :], s2[epo_i, :]) if self.is_epoched else (s1, s2)

            X = symb_symbolise(X, l, m).argsort(kind='quicksort')
            Y = symb_symbolise(Y, l, m).argsort(kind='quicksort')

            # multiply each symbol [1,0,3] by symbol_weights [1,3,9] => [1,0,27] and give a final array of the sum of each code ([.., .., 28, .. ])
            symbol_hash_X = (np.multiply(X, symbol_weights)).sum(1) 
            symbol_hash_Y = (np.multiply(Y, symbol_weights)).sum(1)
    

            p_xy, p_x, p_y = map(symb_normalise_counts, [dict(), dict(), dict()])
            
            for i in range(len(symbol_hash_X)-1):

                xy = f"{symbol_hash_X[i]},{symbol_hash_Y[i]}"
                x,y = str(symbol_hash_X[i]), str(symbol_hash_Y[i])

                for dict_, key in zip([p_xy, p_x, p_y], [xy, x, y]):
                    dict_[key] = dict_.get(key, 0) + 1

            # Normalise counts directly into probabilities
            p_xy, p_x, p_y = [np.array(list(symb_normalise_counts(d).values())) for d in [p_xy, p_x, p_y]]
            
            entropy_X = -np.sum(p_x * np.log2(p_x + np.finfo(float).eps)) 
            entropy_Y = -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
            entropy_XY = -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))

            pairwise[epo_i] = entropy_X + entropy_Y - entropy_XY

        return pairwise


    def __which_mi_estimator(self) -> None:
        """Determines the Mutual Information estimator to be used based on user input. Many estimators are deployed using JIDT."""

        if self.estimator_type == 'histogram':
            self.estimator_name = 'Histogram/Binning Estimator'
            self.calc_sigstats = False # Temporary whilst I figure out how to get p-values for hist/bin estimator
            if self.verbose:
                print("Please not that p-values are not available for Histogram/Binning Estimator as this is not computed using JIDT. Work in progress...")

        elif self.estimator_type == 'ksg1' or self.estimator_type == 'ksg':
            self.estimator_name = 'KSG Estimator (version 1)'
            self.Calc = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1()
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 4)))

        elif self.estimator_type == 'ksg2':
            self.estimator_name = 'KSG Estimator (version 2)'
            self.Calc = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2()
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 4)))
            
        elif self.estimator_type == 'kernel':
            self.estimator_name = 'Box Kernel Estimator'
            self.Calc = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel()
            self.Calc.setProperty("KERNEL_WIDTH", str(self.params.get('kernel_width', 0.25)))

        elif self.estimator_type == 'gaussian':
            self.estimator_name = 'Gaussian Estimator'
            self.Calc = JPackage("infodynamics.measures.continuous.gaussian").MutualInfoCalculatorMultiVariateGaussian()

        elif self.estimator_type == 'symbolic':
            self.estimator_name = 'Symbolic Estimator'
            self.calc_sigstats = False # Temporary whilst I figure out how to get p-values for symbolic estimator
            if self.verbose:
                print("Please not that p-values are not available for Symbolic Estimator as this is not computed using JIDT. Work in progress...")

        else:
            raise ValueError(f"Estimator type {self.estimator_type} not supported. Please choose from 'histogram', 'ksg1', 'ksg2', 'kernel', 'gaussian', 'symbolic'.")

        if not self.estimator_type == 'histogram' and not self.estimator_type == 'symbolic':
            self.Calc.setProperty("NORMALISE", str(self.params.get('normalise', True)))

    def __which_te_estimator(self) -> None:
        """Determines the Transfer Entropy estimator to be used based on user input. Many estimators are deployed using JIDT."""


        if self.estimator_type == 'ksg' or self.estimator_type == 'ksg1' or self.estimator_type == 'ksg2':
            self.estimator_name = 'KSG Estimator'
            self.Calc = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.Calc.setProperty("k_TAU", str(self.params.get('k_tau', 1)))
            self.Calc.setProperty("l_HISTORY", str(self.params.get('l', 1)))
            self.Calc.setProperty("l_TAU", str(self.params.get('l_tau', 1)))
            self.Calc.setProperty("DELAY", str(self.params.get('delay', 1)))
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 4)))
            
        elif self.estimator_type == 'kernel':
            self.estimator_name = 'Box Kernel Estimator'
            self.Calc = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel()
            self.initialise_parameter: Tuple = (self.params.get('k', 1), self.params.get('kernel_width', 0.5))

        elif self.estimator_type == 'gaussian':
            self.estimator_name = 'Gaussian Estimator'
            self.Calc = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.Calc.setProperty("k_TAU", str(self.params.get('k_tau', 1)))
            self.Calc.setProperty("l_HISTORY", str(self.params.get('l', 1)))
            self.Calc.setProperty("l_TAU", str(self.params.get('l_tau', 1)))
            self.Calc.setProperty("DELAY", str(self.params.get('delay', 1)))
            self.Calc.setProperty("BIAS_CORRECTION", str(self.params.get('bias_correction', False)).lower())

        elif self.estimator_type == 'symbolic':
            self.estimator_name = 'Symbolic Estimator'
            self.Calc = JPackage("infodynamics.measures.continuous.symbolic").TransferEntropyCalculatorSymbolic()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.initialise_parameter = (2)

        else:
            raise ValueError(f"Estimator type {self.estimator_type} not supported. Please choose from 'ksg', 'kernel', 'gaussian', or 'symbolic'.")

        self.Calc.setProperty("NORMALISE", str(self.params.get('normalise', True)).lower()) 

    def __estimate_it(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """ Estimates Mutual Information or Transfer Entropy for a pair of time-series signals using JIDT estimators. """

        pairwise = np.zeros((self.n_epo, 4)) # stores MI/TE result, mean, std, p-value

        for epo_i in range(self.n_epo):
            
            X, Y = (s1[epo_i, :], s2[epo_i, :]) if self.is_epoched else (s1, s2)

            self.Calc.initialise(*self.initialise_parameter) if self.initialise_parameter else self.Calc.initialise()
            self.Calc.setObservations(self.__setup_JArray(X), self.__setup_JArray(Y))

            result = self.Calc.computeAverageLocalOfObservations() * np.log(2)

            if self.calc_sigstats:
                stat_sig = self.Calc.computeSignificance(self.stat_sig_perm_num)
                pairwise[epo_i] = [result, stat_sig.getMeanOfDistribution(), stat_sig.getStdOfDistribution(), stat_sig.pValue]
            else:
                pairwise[epo_i, 0] = result
            
        return pairwise
    


    def _plot_it(self, it_matrix: np.ndarray) -> None:
        """Plots the Mutual Information or Transfer Entropy matrix for visualisation. 
        Axes labelled with source and target channel names. 
        Choice to plot for all epochs, specific epoch(s), or average across epochs.

        Args:
            it_matrix (np.ndarray): The Mutual Information or Transfer Entropy matrix to be plotted with shape (n_chan, n_chan, n_epo, 4), 
            where the last dimension represents the statistical signficance testing results: (local result, distribution mean, distribution standard deviation, p-value).
        """

        title = f'{self.measure} | {self.estimator_name} \n {"Inter-Brain" if self.inter_brain else "Intra-Brain"}'
        epochs = [0] # default to un-epoched or epoch-average case
        choice = None
        
        if self.is_epoched: 
            choice = input(f"{self.n_epo} epochs detected. Plot for \n1. All epochs \n2. Specific epoch \n3. Average MI/TE across epochs \nEnter choice: ")
            if choice == "1":
                print("Plotting for all epochs.")
                epochs = range(self.n_epo)
            elif choice == "2":
                epo_choice = input(f"Enter epoch number(s) [1 to {self.n_epo}] separated by comma only: ")
                try:
                    epochs = [int(epo)-1 for epo in epo_choice.split(',')]
                except ValueError:
                    print("Invalid input. Defaulting to plotting all epochs.")
                    epochs = range(self.n_epo)
            elif choice == "3":
                print("Plotting for average MI/TE across epochs. Note that p-values will not be shown.")
                
            else:
                print("Invalid choice. Defaulting to un-epoched data.")

        for epo_i in epochs:
            
            highest = np.max(it_matrix[:,:,epo_i,0])
            channel_pair_with_highest = np.unravel_index(np.argmax(it_matrix[:,:,epo_i,0]), it_matrix[:,:,epo_i,0].shape)
            if self.verbose:
                print(f"Strongest regions: (Source Channel {self.channel_names[0][0][channel_pair_with_highest[0]]} --> " +
                                         f" Target Channel {self.channel_names[1][0][channel_pair_with_highest[1]]}) = {highest}")

            plt.figure(figsize=(12, 10))
            plt.imshow(it_matrix[:,:,epo_i,0], cmap='BuPu', vmin=0, aspect='auto')

            if self.is_epoched and not choice == "3":
                plt.title(f'{title}; Epoch {epo_i+1}', pad=20)
            else:
                plt.title(title, pad=20)
                
            if self.calc_sigstats and not choice == "3" and not self.estimator_type == 'histogram' and not self.estimator_type == 'symbolic': # Again, temporary.
                for i in range(self.n_chan):
                    for j in range(self.n_chan):
                        p_val = float(it_matrix[i, j, epo_i, 3])
                        if p_val < self.p_threshold and (not self.inter_brain and i != j):
                            normalized_value = (it_matrix[i, j, epo_i, 0] - np.min(it_matrix[:,:,epo_i,0])) / (np.max(it_matrix[:,:,epo_i,0]) - np.min(it_matrix[:,:,epo_i,0]))
                            text_colour = 'white' if normalized_value > 0.5 else 'black'
                            plt.text(j, i, f'p={p_val:.2f}', ha='center', va='center', color=text_colour, fontsize=8, fontweight='bold')

            plt.colorbar()
            plt.xlabel('Target Channels')
            plt.ylabel('Source Channels')
            plt.xticks(range(self.n_chan), self.channel_names[1][0], rotation=90) 
            plt.yticks(range(self.n_chan), self.channel_names[0][0])
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labeltop=True)
            plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
            plt.show()

    @staticmethod
    def _plot_atoms(phi_dict: dict, channel_indices: list):

        ch_X, ch_Y = channel_indices
        value_dict = phi_dict[ch_X][ch_Y]

        image = Image.open('visualisations/atoms_lattice_values.png')
        draw = ImageDraw.Draw(image) 

        text_positions = {
            'rtr': (485, 1007), 
            'rtx': (160, 780),
            'rty': (363, 780), 
            'rts': (37, 512), 
            'xtr': (610, 779), 
            'xtx': (237, 510), 
            'xty': (487, 585), 
            'xts': (160, 243), 
            'ytr': (800, 780), 
            'ytx': (485, 427), 
            'yty': (725, 505), 
            'yts': (363, 243), 
            'str': (930, 505), 
            'stx': (605, 243), 
            'sty': (807, 243), 
            'sts': (485, 41)   
        }

        for text, pos in text_positions.items():
            value = value_dict.get(text, '')
            plot_text = f"{round(float(value), 3):.3f}"
            draw.text(pos, plot_text, fill="black", font_size=25)

        image.show()


    def compute_mi(self, estimator_type: str = 'kernel', calc_sigstats: bool = False, vis: bool = False, **kwargs) -> np.ndarray:
        """Function to compute Mutual Information between data (time-series signals) instantiated in the HyperIT object.

        Args:
            estimator_type       (str, optional): Which Mutual Information estimator to use. Defaults to 'kernel'.
            calc_sigstats       (bool, optional): Whether to conduct statistical signficance testing. Defaults to False.
            vis                 (bool, optional): Whether to visualise (via _plot_it()). Defaults to False.

        Returns:
                                    (np.ndarray): A matrix of Mutual Information values with shape (n_chan, n_chan, n_epo, 4),
                                                  where the last dimension represents the statistical signficance testing results: (local result, distribution mean, distribution standard deviation, p-value). 
                                                  If calc_sigstats is False, only the local results will be returned in this last dimension.
        """
        
        self.measure = 'Mutual Information'
        self.estimator_type: str = estimator_type.lower()
        self.calc_sigstats: bool = calc_sigstats
        self.vis: bool = vis
        self.params = kwargs

        self.stat_sig_perm_num = self.params.get('stat_sig_perm_num', 100)
        self.p_threshold = self.params.get('p_threshold', 0.05)
        
        self.estimator = self.__which_mi_estimator()

        if self.estimator_type == 'histogram' or self.estimator_type == 'symbolic':
            self.mi_matrix = np.zeros((self.n_chan, self.n_chan, self.n_epo, 1)) # TEMPORARY, until I figure out how to get p-values for hist/bin and symbolic MI estimators
        
        else:
            self.mi_matrix = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))
        

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):

                if self.inter_brain or i != j:
                    s1, s2 = (self.data1[:, i, :], self.data2[:, j, :]) if self.is_epoched else (self.data1[i, :], self.data2[j, :])
                    
                    if self.estimator_type == 'histogram':
                        self.mi_matrix[i, j] = self._mi_hist(s1, s2)
                    elif self.estimator_type == 'symbolic':
                        self.mi_matrix[i, j] = self._mi_symb(s1, s2)
                    else:
                        self.mi_matrix[i, j] = self.__estimate_it(s1, s2)

                    if not self.inter_brain:
                        self.mi_matrix[j, i] = self.mi_matrix[i, j]

        mi = np.array((self.mi_matrix))

        if self.vis:
            self._plot_it(mi)

        return mi
    
    def compute_te(self, estimator_type: str = 'kernel', calc_sigstats: bool = False, vis: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Function to compute Transfer Entropy between data (time-series signals) instantiated in the HyperIT object. 
            data1 is first taken to be the source and data2 the target (X->Y). This function automatically computes the opposite matrix for Y -> X.

        Args:
            estimator_type       (str, optional): Which Mutual Information estimator to use. Defaults to 'kernel'.
            calc_sigstats       (bool, optional): Whether to conduct statistical signficance testing. Defaults to False.
            vis                 (bool, optional): Whether to visualise (via _plot_it()). Defaults to False.

        Returns:
                   Tuple(np.ndarray, np.ndarray): Two matrices of Transfer Entropy values (X->Y and Y->X), each with shape (n_chan, n_chan, n_epo, 4),
                                                  where the last dimension represents the statistical signficance testing results: (local result, distribution mean, distribution standard deviation, p-value). 
                                                  If calc_sigstats is False, only the local results will be returned in this last dimension.
        """
        
        self.measure = 'Transfer Entropy'
        self.estimator_type: str = estimator_type.lower()
        self.calc_sigstats: bool = calc_sigstats
        self.vis: bool = vis
        self.params = kwargs
        
        self.stat_sig_perm_num = self.params.get('stat_sig_perm_num', 100)
        self.p_threshold = self.params.get('p_threshold', 0.05)

        self.estimator = self.__which_te_estimator()

        self.te_matrix_xy = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))
        self.te_matrix_yx = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):
                
                if self.inter_brain or i != j: # avoid self-channel calculations for intra_brain condition
                    
                    s1, s2 = (self.data1[:, i, :], self.data2[:, j, :]) if self.is_epoched else (self.data1[i, :], self.data2[j, :]) # whether to keep epochs
                    self.te_matrix_xy[i, j] = self.__estimate_it(s1, s2)
                    
                    if self.inter_brain: # don't need to compute opposite matrix for intra-brain as we already loop through each channel combination including symmetric
            
                        self.te_matrix_yx[i, j] = self.__estimate_it(s2, s1)
                    
        te_xy = np.array((self.te_matrix_xy))
        te_yx = np.array((self.te_matrix_yx))
        
        if self.vis:
            print("Plotting Transfer Entropy for X -> Y...")
            self._plot_it(te_xy)
            if self.inter_brain:
                print("Plotting Transfer Entropy for Y -> X...")
                self._plot_it(te_yx)
                
        return te_xy, te_yx

    def compute_atoms(self, tau: int = 1, kind: str = "gaussian", redundancy: str = "MMI", vis: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Function to compute Integrated Information Decomposition (ΦID) between data (time-series signals) instantiated in the HyperIT object.
            Option to visualise the lattice values for a specific channel pair (be sure to specify via plot_channels kwarg).

        Args:
            tau             (int, optional): The time-lag parameter for the ΦID calculation. Defaults to 1.
            kind            (str, optional): The estimator to be used for the ΦID calculation. Defaults to "gaussian".
            redundancy      (str, optional): The redundancy measure to be used for the ΦID calculation. Defaults to "MMI".
            vis            (bool, optional): Whether to visualise (via _plot_atoms()). Defaults to False.

        Returns:
              Tuple(np.ndarray, np.ndarray): Two matrices of Integrated Information Decomposition dictionaries (representing all atoms, both X->Y and Y->X), each with shape (n_chan, n_chan),
        """
        
        self.measure = 'Integrated Information Decomposition'
        self.tau = tau
        self.kind = kind
        self.redundancy = redundancy

        phi_dict_xy = [[{} for _ in range(self.n_chan)] for _ in range(self.n_chan)]
        phi_dict_yx = [[{} for _ in range(self.n_chan)] for _ in range(self.n_chan)]

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):
                
                if self.inter_brain or i != j:
                    s1, s2 = (self.data1[:, i, :], self.data2[:, j, :]) if self.is_epoched else (self.data1[i, :], self.data2[j, :])
                    
                    atoms_results, _ = calc_PhiID(s1, s2, tau=self.tau, kind=self.kind, redundancy=self.redundancy)
                    calc_atoms = np.mean(np.array([atoms_results[_] for _ in PhiID_atoms_abbr]), axis=1)
                    phi_dict_xy[i][j] = {key: value for key, value in zip(atoms_results.keys(), calc_atoms)}

                    if self.inter_brain:
                        atoms_results, _ = calc_PhiID(s2, s1, tau=self.tau, kind=self.kind, redundancy=self.redundancy)
                        calc_atoms = np.mean(np.array([atoms_results[_] for _ in PhiID_atoms_abbr]), axis=1)
                        phi_dict_yx[i][j] = {key: value for key, value in zip(atoms_results.keys(), calc_atoms)}   

        if vis:
            plot_channels = kwargs.get('plot_channels', [0, 0])
            print(plot_channels)
            self._plot_atoms(phi_dict_xy, plot_channels)

        return phi_dict_xy, phi_dict_yx




if __name__ == "__main__":

    np.random.seed(42)
    n = 1000

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)
    E = np.zeros(n)
    F = np.zeros(n)
    A[0], B[0], C[0], D[0], E[0], F[0] = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

    std_dev = 0.1
    for t in range(1, n):
        A[t] = np.sin(C[t-1] + E[t-1]) + np.random.normal(0, std_dev)
        B[t] = 0.5 * A[t-1] + np.random.normal(0, std_dev)
        C[t] = 0.3 * B[t-1] + np.exp(-D[t-1]) + np.random.normal(0, std_dev)
        D[t] = D[t-1]**2 - 0.1 * D[t-1] + np.random.normal(0, std_dev)
        E[t] = 0.7 * B[t-1] + np.cos(A[t-1]) + np.random.normal(0, std_dev)
        F[t] = 3 * np.sin(F[t-1]) + np.random.normal(0, std_dev)
        
    # def epoch_it(data, n_epochs):
    #     if len(data) % n_epochs != 0:
    #         raise ValueError("The length of the time series must be divisible by the number of epochs.")
    #     epoch_length = len(data) // n_epochs
    #     return data.reshape(n_epochs, epoch_length)

    # A = epoch_it(A, 10)
    # B = epoch_it(B, 10)
    # C = epoch_it(C, 10)
    # D = epoch_it(D, 10)
    # E = epoch_it(E, 10)
    # F = epoch_it(F, 10)

    ### FOR EPOCHED DATA
    # data1 = np.stack([A, B, C], axis = 1) #  10, 3, 100 (epo, ch, sample)
    # data2 = np.stack([D, E, F], axis = 1) #  10, 3, 100 (epo, ch, sample)

    ### FOR UNEPOCHED DATA
    #data1 = np.vstack([A, B, C]) #  3, 1000 (ch, sample)
    #data = np.vstack([D, E, F])  #  3, 1000 (ch, sample)

    

    data = np.vstack([A, B, C, D, E, F])
    channel_names = [['A', 'B', 'C', 'D', 'E', 'F'], ['A', 'B', 'C', 'D', 'E', 'F']]

    it = HyperIT(data, data, channel_names=channel_names)


    plot_channels = [0,0]
    phi_dict_xy, phi_dict_yx = it.compute_atoms(vis=True, plot_channels=plot_channels)

    # it.compute_mi(estimator_type='symbolic', calc_sigstats=False, vis=True)
    # it.compute_te(estimator_type='gaussian', calc_sigstats=True, vis=True)
    
    #te_matrix_xy, te_matrix_yx = it.compute_te()
