import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import mode
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from jpype import *



class HyperIT(ABC):
    def __init__(self, measure: str, eeg1: np.ndarray, eeg2: np.ndarray, channel_names: List[str], calc_sigstats: bool, vis: bool, **kwargs):
        self.measure: str = measure
        self.eeg1: np.ndarray = eeg1
        self.eeg2: np.ndarray = eeg2
        self.channel_names: List[str] = channel_names
        self.inter_brain: bool = not np.array_equal(eeg1, eeg2)
        self.is_epoched: bool = (eeg1.ndim == 3)

        self.check_eeg_data()

        if self.is_epoched:
            self.n_epo, self.n_chan, self.n_samples = self.eeg1.shape
        else:
            self.n_epo = 1
            self.n_chan, self.n_samples = self.eeg1.shape
        
        self.vis: bool = vis
        self.params = kwargs
        self.estimator_name: str = None
        self.calc_sigstats: bool = calc_sigstats
        self.p_threshold = 0.05

        if self.measure == "MI":
            self.estimator = self.which_mi_estimator(**kwargs)
        elif self.measure == "TE":
            self.estimator = self.which_te_estimator(**kwargs)

    def check_eeg_data(self):

        assert self.measure == "MI" or self.measure == "TE", "Measure must be either 'MI' or 'TE'."

        if not isinstance(self.eeg1, np.ndarray) or not isinstance(self.eeg2, np.ndarray):
            raise ValueError("EEG data must be a numpy array.")
        
        assert self.eeg1.shape == self.eeg2.shape, "Data passed should have the same number of epochs for each participant."

        if self.eeg1.ndim not in [2,3]:
            raise ValueError(f"The EEG signals passed do not have the correct shape. Expected 2 dimensions (n_chan, time_points) or 3 dimensions (n_epochs, n_chan, time_points); instead, received {self.eeg1.ndim}.")

        if isinstance(self.channel_names, str):
            raise ValueError("Channel names must be a list of strings.")
        if not isinstance(self.channel_names[0], list) or not self.inter_brain: # if not list of lists (i.e., only one list provided,therefore intrabrain), then double channel names.
            self.channel_names = [self.channel_names, self.channel_names]
        
        if self.is_epoched:
            assert self.eeg1.shape[1] == len(self.channel_names[0][0]), "The number of channels in eeg1 does not match the length of channel_names[0]"
            assert self.eeg2.shape[1] == len(self.channel_names[1][0]), "The number of channels in eeg2 does not match the length of channel_names[1]"
        else:
            assert self.eeg1.shape[0] == len(self.channel_names[0][0]), "The number of channels in eeg1 does not match the length of channel_names[0]"
            assert self.eeg2.shape[0] == len(self.channel_names[1][0]), "The number of channels in eeg2 does not match the length of channel_names[1]"
        
    @staticmethod
    def setup_JIDT(working_directory: str):
        if(not isJVMStarted()):
            jarLocation = os.path.join(working_directory, "..", "..", "infodynamics.jar")
            # Usually, just specifying the current working directory (cwd) would suffice; if not, use specific location, e.g., below
            jarLocation = "/Users/edoardochidichimo/Desktop/MPhil_Code/IT-Hyperscanning/infodynamics.jar"

            if (not(os.path.isfile(jarLocation))):
                exit("infodynamics.jar not found (expected at " + os.path.abspath(jarLocation) + ") - are you running from demos/python?")

            startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
        
        else:
            exit("JVM has already started. Please exit by running shutdownJVM() in your Python console.")

    @staticmethod
    def setup_JArray(a: np.ndarray) -> JArray:
        a = (a).astype(np.float64) 
        try:
            ja = JArray(JDouble, 1)(a)
        except Exception: 
            ja = JArray(JDouble, 1)(a.tolist())
        return ja
    
    @staticmethod
    def calc_fd_bins(X: np.array, Y: np.array) -> int:
        X = X.flatten()
        Y = Y.flatten()
        # Freedman-Diaconis Rule for frequency-distribution bin size
        fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
        fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
        fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
        return fd_bins
        
    def estimate_it(self, s1: np.ndarray, s2: np.ndarray):
        pairwise = np.zeros((self.n_epo, 4))

        for epo_i in range(self.n_epo):
            
            X, Y = (s1[epo_i, :], s2[epo_i, :]) if self.is_epoched else (s1, s2)
            
            self.Calc.initialise()
            self.Calc.setObservations(self.setup_JArray(X), self.setup_JArray(Y))

            result = self.Calc.computeAverageLocalOfObservations() * np.log(2)

            if self.calc_sigstats:
                stat_sig = self.Calc.computeSignificance(self.stat_sig_perm_num)
                distr_mean = stat_sig.getMeanOfDistribution()
                distr_std = stat_sig.getStdOfDistribution()
                p_val = stat_sig.pValue
            else:
                distr_mean, distr_std, p_val = None, None, None
            
            pairwise[epo_i] = [result, distr_mean, distr_std, p_val]

        return pairwise
    
    def plot_it(self, it_matrix: np.ndarray):
        """Plots heatmap of mutual information or transfer entropy values for either intra-brain or inter-brain design.

        Args:
            it_matrix (np.ndarray): Matrix with shape (n_chan1, n_chan2). Can be same channels (intra-brain) or two-person channel (inter-brain). Note that intrabrain MI will be a symmetric heatmap. 
            inter_brain (bool): Whether the analysis is inter-brain analysis or not (i.e., intra-brain)
            channel_names (List[str]): List of channel names of EEG signals. Either a single list for intra-brain or two lists for inter-brain. 
        """

        title = f'Inter-Brain // {self.estimator_name}' if self.inter_brain else f'Intra-Brain // {self.estimator_name}'
        choice = None

        ## check if sigstats is just a single value or a matrix (i.e., un-epoched or epochs maintained). If matrix, ask if to plot for every epoch or specific epochs.
        if self.is_epoched: 
            print(f"{self.n_epo} epochs detected. Do you want to plot for all epochs or specific epochs?")
            print("1. All epochs")
            print("2. Specific epoch")
            print("3. Average MI/TE across epochs")
            choice = input("Enter choice: ")
            if choice == "1":
                print("Plotting for all epochs.")
                epochs = range(self.n_epo)
            elif choice == "2":
                print(f"Choose from 1 to {self.n_epo}")
                epo_choice = input("Enter epoch number(s): ")
                selected_epochs = np.array([int(num.strip()) for num in epo_choice.replace(',', ' ').split()])
                try:
                    epochs = [int(epo-1) for epo in selected_epochs]
                except ValueError:
                    print("All epochs must be integer numbers. Please try again.")
            elif choice == "3":
                print("Plotting for average MI/TE across epochs. Note that p-values will not be shown.")
                epochs = [0]

            else:
                raise ValueError("Invalid choice. Please enter 1 or 2.")
            
        else:
            print("Un-epoched data detected. Plotting for un-epoched data.")
            epochs = [0]

        for epo_i in epochs:

            plt.matshow(it_matrix[:,:,epo_i,0], cmap='BuPu', vmin=0)

            if self.is_epoched and not choice == "3":
                plt.title(f'{title}; Epoch {epo_i+1}', pad=20)
            else:
                plt.title(title, pad=20)
                
            if self.calc_sigstats and not choice == "3":
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
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)

            highest = np.max(it_matrix[:,:,epo_i,0])
            channel_pair_with_highest = np.unravel_index(np.argmax(it_matrix[:,:,epo_i,0]), it_matrix[:,:,epo_i,0].shape)
            print(f"Strongest regions: (Source Channel {self.channel_names[0][0][channel_pair_with_highest[0]]} --> " +
                                    f" Target Channel {self.channel_names[1][0][channel_pair_with_highest[1]]}) = {highest}")
            
            plt.show()


    def which_mi_estimator(self, **kwargs):

        self.stat_sig_perm_num = kwargs.get('stat_sig_perm_num', 100)

        if self.estimator_type == 'histogram':
            self.estimator_name = 'Histogram/Binning Estimator'
            self.calc_sigstats = False # Temporary whilst I figure out how to get p-values for hist/bin estimator
            print("Please not that p-values are not available for Histogram/Binning Estimator as this is not computed using JIDT. Work in progress...")

        elif self.estimator_type == 'ksg1' or self.estimator_type == 'ksg':
            self.estimator_name = 'KSG Estimator (version 1)'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 1)))

        elif self.estimator_type == 'ksg2':
            self.estimator_name = 'KSG Estimator (version 2)'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 1)))
            
        elif self.estimator_type == 'kernel':
            self.estimator_name = 'Box Kernel Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
            self.Calc = self.CalcClass()
            self.Calc.setProperty("NORMALISE", "true") 
            self.Calc.setProperty("KERNEL_WIDTH", str(self.params.get('kernel_width', 0.25)))

        elif self.estimator_type == 'gaussian':
            self.estimator_name = 'Gaussian Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.gaussian").MutualInfoCalculatorMultiVariateGaussian
            self.Calc = self.CalcClass()

        elif self.estimator_type == 'symbolic':
            self.estimator_name = 'Symbolic Estimator'
            self.calc_sigstats = False # Temporary whilst I figure out how to get p-values for symbolic estimator
            print("Please not that p-values are not available for Symbolic Estimator as this is not computed using JIDT. Work in progress...")

        else:
            raise ValueError(f"Estimator type {self.estimator_type} not supported. Please choose from 'histogram', 'ksg1', 'ksg2', 'kernel', 'gaussian', 'symbolic'.")

    def which_te_estimator(self, **kwargs):

        self.stat_sig_perm_num = kwargs.get('stat_sig_perm_num', 100)

        if self.estimator_type == 'ksg':
            self.estimator_name = 'KSG Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.Calc.setProperty("k_TAU", str(self.params.get('k_tau', 1)))
            self.Calc.setProperty("l_HISTORY", str(self.params.get('l', 1)))
            self.Calc.setProperty("l_TAU", str(self.params.get('l_tau', 1)))
            self.Calc.setProperty("DELAY", str(self.params.get('delay', 1)))
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 1)))
            
        elif self.estimator_type == 'kernel':
            self.estimator_name = 'Box Kernel Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
            self.Calc = self.CalcClass()
            self.Calc.setProperty("NORMALISE", "true") 
            self.initialise_parameter: Tuple = (self.params.get('k', 1), self.params.get('kernel_width', 0.5))

        elif self.estimator_type == 'gaussian':
            self.estimator_name = 'Gaussian Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.Calc.setProperty("k_TAU", str(self.params.get('k_tau', 1)))
            self.Calc.setProperty("l_HISTORY", str(self.params.get('l', 1)))
            self.Calc.setProperty("l_TAU", str(self.params.get('l_tau', 1)))
            self.Calc.setProperty("DELAY", str(self.params.get('delay', 1)))
            self.Calc.setProperty("BIAS_CORRECTION", str(self.params.get('bias_correction', False)).lower())

        elif self.estimator_type == 'symbolic':
            self.estimator_name = 'Symbolic Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.symbolic").TransferEntropyCalculatorSymbolic
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.initialise_parameter = (2)

        else:
            raise ValueError(f"Estimator type {self.estimator_type} not supported. Please choose from 'ksg', 'kernel', 'gaussian', or 'symbolic'.")


    def compute_mi(self):

        self.mi_matrix = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):

                if self.inter_brain or i != j:
                    s1, s2 = (self.eeg1[:, i, :], self.eeg2[:, j, :]) if self.is_epoched else (self.eeg1[i, :], self.eeg2[j, :])
                    
                    self.mi_matrix[i, j] = self.estimate_it(s1, s2)

                    if not self.inter_brain:
                        self.mi_matrix[j, i] = self.mi_matrix[i, j]

        mi = np.array((self.mi_matrix))

        if self.vis:
            self.plot_it(mi)

        return mi

    def compute_te(self):

        self.te_matrix_xy = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))
        self.te_matrix_yx = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):
                
                if self.inter_brain or i != j: # avoid self-channel calculations for intra_brain condition
                    
                    s1, s2 = (self.eeg1[:, i, :], self.eeg2[:, j, :]) if self.is_epoched else (self.eeg1[i, :], self.eeg2[j, :]) # whether to keep epochs
                    self.te_matrix_xy[i, j] = self.estimate_it(s1, s2)
                    
                    if self.inter_brain: # don't need to compute opposite matrix for intra-brain as we already loop through each channel combination including symmetric
            
                        self.te_matrix_yx[i, j] = self.estimate_it(s2, s1)
                    
        te_xy = np.array((self.te_matrix_xy))
        te_yx = np.array((self.te_matrix_yx))
        
        if self.vis:
            print("Plotting Transfer Entropy for X -> Y...")
            self.plot_it(te_xy)
            if self.inter_brain:
                print("Plotting Transfer Entropy for Y -> X...")
                self.plot_it(te_yx)
                
        return te_xy, te_yx









class MI(HyperIT):
    
    def __init__(self, eeg1: np.ndarray, eeg2: np.ndarray, channel_names: List[str], calc_sigstats: bool, estimator_type: str, **kwargs):
        super().__init__(eeg1, eeg2, channel_names, calc_sigstats, **kwargs)
        self.setup_JIDT(os.getcwd())
        self.estimator_type = estimator_type
        self.estimator = self.which_mi_estimator(**kwargs)


    def which_mi_estimator(self, **kwargs):

        self.stat_sig_perm_num = kwargs.get('stat_sig_perm_num', 100)

        if self.estimator_type == 'histogram':
            self.estimator_name = 'Histogram/Binning Estimator'
            self.calc_sigstats = False # Temporary whilst I figure out how to get p-values for hist/bin estimator
            print("Please not that p-values are not available for Histogram/Binning Estimator as this is not computed using JIDT. Work in progress...")

        elif self.estimator_type == 'ksg1':
            self.estimator_name = 'KSG Estimator (version 1)'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 1)))

        elif self.estimator_type == 'ksg2':
            self.estimator_name = 'KSG Estimator (version 2)'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 1)))
            
        elif self.estimator_type == 'kernel':
            self.estimator_name = 'Box Kernel Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
            self.Calc = self.CalcClass()
            self.Calc.setProperty("NORMALISE", "true") 
            self.Calc.setProperty("KERNEL_WIDTH", str(self.params.get('kernel_width', 0.25)))

        elif self.estimator_type == 'gaussian':
            self.estimator_name = 'Gaussian Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.gaussian").MutualInfoCalculatorMultiVariateGaussian
            self.Calc = self.CalcClass()

        elif self.estimator_type == 'symbolic':
            self.estimator_name = 'Symbolic Estimator'
            self.calc_sigstats = False # Temporary whilst I figure out how to get p-values for symbolic estimator
            print("Please not that p-values are not available for Symbolic Estimator as this is not computed using JIDT. Work in progress...")

        else:
            raise ValueError(f"Estimator type {self.estimator_type} not supported. Please choose from 'histogram', 'ksg1', 'ksg2', 'kernel', 'gaussian', 'symbolic'.")


    @staticmethod
    def calc_fd_bins(X: np.array, Y: np.array) -> int:
        X = X.flatten()
        Y = Y.flatten()
        # Freedman-Diaconis Rule for frequency-distribution bin size
        fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
        fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
        fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
        return fd_bins

    def compute_mi(self):

        self.mi_matrix = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):

                if self.inter_brain or i != j:
                    s1, s2 = (self.eeg1[:, i, :], self.eeg2[:, j, :]) if self.is_epoched else (self.eeg1[i, :], self.eeg2[j, :])
                    
                    self.mi_matrix[i, j] = self.estimate_it(s1, s2)

                    if not self.inter_brain:
                        self.mi_matrix[j, i] = self.mi_matrix[i, j]

        mi = np.array((self.mi_matrix))

        if self.vis:
            self.plot_it(mi)

        return mi


class TE(HyperIT):

    def __init__(self, eeg1: np.ndarray, eeg2: np.ndarray, channel_names: List[str], calc_sigstats: bool, estimator_type: str, **kwargs):
        super().__init__(eeg1, eeg2, channel_names, calc_sigstats, **kwargs)
        self.setup_JIDT(os.getcwd())
        self.estimator_type = estimator_type
        self.estimator = self.which_te_estimator(**kwargs)

    def which_te_estimator(self, **kwargs):

        self.stat_sig_perm_num = kwargs.get('stat_sig_perm_num', 100)

        if self.estimator_type == 'ksg':
            self.estimator_name = 'KSG Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.Calc.setProperty("k_TAU", str(self.params.get('k_tau', 1)))
            self.Calc.setProperty("l_HISTORY", str(self.params.get('l', 1)))
            self.Calc.setProperty("l_TAU", str(self.params.get('l_tau', 1)))
            self.Calc.setProperty("DELAY", str(self.params.get('delay', 1)))
            self.Calc.setProperty("k", str(self.params.get('kraskov_param', 1)))
            
        elif self.estimator_type == 'kernel':
            self.estimator_name = 'Box Kernel Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
            self.Calc = self.CalcClass()
            self.Calc.setProperty("NORMALISE", "true") 
            self.initialise_parameter: Tuple = (self.params.get('k', 1), self.params.get('kernel_width', 0.5))

        elif self.estimator_type == 'gaussian':
            self.estimator_name = 'Gaussian Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.Calc.setProperty("k_TAU", str(self.params.get('k_tau', 1)))
            self.Calc.setProperty("l_HISTORY", str(self.params.get('l', 1)))
            self.Calc.setProperty("l_TAU", str(self.params.get('l_tau', 1)))
            self.Calc.setProperty("DELAY", str(self.params.get('delay', 1)))
            self.Calc.setProperty("BIAS_CORRECTION", str(self.params.get('bias_correction', False)).lower())

        elif self.estimator_type == 'symbolic':
            self.estimator_name = 'Symbolic Estimator'
            self.CalcClass = JPackage("infodynamics.measures.continuous.symbolic").TransferEntropyCalculatorSymbolic
            self.Calc = self.CalcClass()
            self.Calc.setProperty("k_HISTORY", str(self.params.get('k', 1)))
            self.initialise_parameter = (2)

        else:
            raise ValueError(f"Estimator type {self.estimator_type} not supported. Please choose from 'ksg', 'kernel', 'gaussian', or 'symbolic'.")

    def compute_te(self):

        self.te_matrix_xy = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))
        self.te_matrix_yx = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):
                
                if self.inter_brain or i != j: # avoid self-channel calculations for intra_brain condition
                    
                    s1, s2 = (self.eeg1[:, i, :], self.eeg2[:, j, :]) if self.is_epoched else (self.eeg1[i, :], self.eeg2[j, :]) # whether to keep epochs
                    self.te_matrix_xy[i, j] = self.estimate_it(s1, s2)
                    
                    if self.inter_brain: # don't need to compute opposite matrix for intra-brain as we already loop through each channel combination including symmetric
            
                        self.te_matrix_yx[i, j] = self.estimate_it(s2, s1)
                    
        te_xy = np.array((self.te_matrix_xy))
        te_yx = np.array((self.te_matrix_yx))
        
        if self.vis:
            print("Plotting Transfer Entropy for X -> Y...")
            self.plot_it(te_xy)
            if self.inter_brain:
                print("Plotting Transfer Entropy for Y -> X...")
                self.plot_it(te_yx)
                
        return te_xy, te_yx















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
    
        
        
    def epoch_it(data, n_epochs):
        if len(data) % n_epochs != 0:
            raise ValueError("The length of the time series must be divisible by the number of epochs.")
        epoch_length = len(data) // n_epochs
        return data.reshape(n_epochs, epoch_length)


    # A = epoch_it(A, 10)
    # B = epoch_it(B, 10)
    # C = epoch_it(C, 10)
    # D = epoch_it(D, 10)
    # E = epoch_it(E, 10)
    # F = epoch_it(F, 10)

    # eeg_data1 = np.stack([A, B, C], axis = 1) # EPOCHED 10, 3, 100 (epo, ch, sample)
    # eeg_data2 = np.stack([D, E, F], axis = 1) # EPOCHED 10, 3, 100 (epo, ch, sample)

    #eeg_data1 = np.vstack([A, B, C]) # UNEPOCHED 3, 1000 (ch, sample)
    #eeg_data = np.vstack([D, E, F]) # UNEPOCHED 3, 1000 (ch, sample)


    eeg_data = np.vstack([A, B, C, D, E, F])
    print(eeg_data.shape)
    channel_names = [['A', 'B', 'C', 'D', 'E', 'F'], ['A', 'B', 'C', 'D', 'E', 'F']]

    te = HyperIT("TE", eeg_data, eeg_data, channel_names=channel_names, calc_sigstats=True, estimator_type='ksg', vis=True)
    te_matrix_xy, te_matrix_yx = te.compute_te()