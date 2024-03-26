import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from jpype import *


class HyperIT(ABC):
    def __init__(self, eeg1: np.ndarray, eeg2: np.ndarray, channel_names: List[str], calc_sigstats: bool, estimator_type: str, vis: bool, **kwargs):
        
        self.setup_JIDT(os.getcwd())
        
        self.eeg1: np.ndarray = eeg1
        self.eeg2: np.ndarray = eeg2
        self.channel_names: List[str] = channel_names
        self.inter_brain: bool = not np.array_equal(eeg1, eeg2)
        self.is_epoched: bool = eeg1.ndim == 3
        self.params = kwargs

        self.check_eeg_data()

        if self.is_epoched:
            self.n_epo, self.n_chan, self.n_samples = self.eeg1.shape
        else:
            self.n_epo = 1
            self.n_chan, self.n_samples = self.eeg1.shape
        
        self.vis: bool = vis
        self.estimator_type: str = estimator_type
        self.calc_sigstats: bool = calc_sigstats
        self.stat_sig_perm_num = self.params.get('stat_sig_perm_num', 100)
        self.p_threshold = self.params.get('p_threshold', 0.05)

    def check_eeg_data(self):

        if not all(isinstance(eeg, np.ndarray) for eeg in [self.eeg1, self.eeg2]):
            raise ValueError("Data must be numpy arrays.")
        
        if self.eeg1.shape != self.eeg2.shape:
            raise ValueError("Data must have the same shape for both participants.")
    
        if self.eeg1.ndim not in [2,3]:
            raise ValueError(f"Unexpected number of dimensions in EEG data: {self.eeg1.ndim}. Expected 2 dimensions (channels, time_points) or 3 dimensions (epochs, channels, time_points); instead, received {self.eeg1.ndim}.")

        if not isinstance(self.channel_names, (list, np.ndarray)) or isinstance(self.channel_names[0], str):
            raise ValueError("Channel names must be a list of strings or a list of lists of strings for inter-brain analysis.")
    
        if not self.inter_brain and isinstance(self.channel_names[0], list):
            self.channel_names = [self.channel_names] * 2

        n_channels = self.eeg1.shape[1] if self.is_epoched else self.eeg1.shape[0]
        if any(len(names) != n_channels for names in self.channel_names[0]):
            raise ValueError("The number of channels in EEG data does not match the length of channel_names.")
    
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
        
    def mi_hist(self, s1: np.ndarray, s2: np.ndarray) -> float:

        @staticmethod
        def calc_fd_bins(X: np.ndarray, Y: np.ndarray) -> int:

            # Freedman-Diaconis Rule for frequency-distribution bin size
            fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
            fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
            fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
            # print("Optimal frequency-distribution bin size for histogram estimator (à la Freedman-Diaconis Rule): ", fd_bins)
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

    def mi_symb(self, s1: np.ndarray, s2: np.ndarray, l: int = 1, m: int = 3) -> float:

        hashmult = np.power(m, np.arange(m))
        pairwise = np.zeros((self.n_epo, 1))

        @staticmethod
        def symb_symbolise(X: np.ndarray, l: int, m: int) -> np.ndarray:
            Y = np.empty((m, len(X) - (m - 1) * l))
            for i in range(m):
                Y[i] = X[i * l:i * l + Y.shape[1]]
            return Y.T

        @staticmethod   
        def symb_incr_counts(key,d) -> None:
            d[key] = d.get(key, 0) + 1

        @staticmethod
        def symb_normalise(d) -> None:
            s=sum(d.values())        
            for key in d:
                d[key] /= s
        
        for epo_i in range(self.n_epo):

            X, Y = (s1[epo_i, :], s2[epo_i, :]) if self.is_epoched else (s1, s2)

            X = symb_symbolise(X, l, m).argsort(kind='quicksort')
            Y = symb_symbolise(Y, l, m).argsort(kind='quicksort')

            hashval_X = (np.multiply(X, hashmult)).sum(1) # multiply each symbol [1,0,3] by hashmult [1,3,9] => [1,0,27] and give a final array of the sum of each code ([.., .., 28, .. ])
            hashval_Y = (np.multiply(Y, hashmult)).sum(1)
            
            x_sym_to_perm = hashval_X
            y_sym_to_perm = hashval_Y
            
            p_xy = {}
            p_x = {}
            p_y = {}
            
            for i in range(len(x_sym_to_perm)-1):
                xy = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i])
                x = str(x_sym_to_perm[i])
                y = str(y_sym_to_perm[i])
                symb_incr_counts(xy,p_xy)
                symb_incr_counts(x,p_x)
                symb_incr_counts(y,p_y)
                
            symb_normalise(p_xy)
            symb_normalise(p_x)
            symb_normalise(p_y)
            
            p_xy = np.array(list(p_xy.values()))
            p_x = np.array(list(p_x.values()))
            p_y = np.array(list(p_y.values()))
            
            entropy_X = -np.sum(p_x * np.log2(p_x + np.finfo(float).eps)) 
            entropy_Y = -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
            entropy_XY = -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))

            result = entropy_X + entropy_Y - entropy_XY

            pairwise[epo_i] = result

        return pairwise


    def which_mi_estimator(self):

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

    def which_te_estimator(self):

        if self.estimator_type == 'ksg' or self.estimator_type == 'ksg1' or self.estimator_type == 'ksg2':
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


    def estimate_it(self, s1: np.ndarray, s2: np.ndarray):
        
        pairwise = np.zeros((self.n_epo, 4)) # stores MI/TE result, mean, std, p-value

        for epo_i in range(self.n_epo):
            
            X, Y = (s1[epo_i, :], s2[epo_i, :]) if self.is_epoched else (s1, s2)
            
            self.Calc.initialise()
            self.Calc.setObservations(self.setup_JArray(X), self.setup_JArray(Y))

            result = self.Calc.computeAverageLocalOfObservations() * np.log(2)

            if self.calc_sigstats:
                stat_sig = self.Calc.computeSignificance(self.stat_sig_perm_num)
                pairwise[epo_i] = [result, stat_sig.getMeanOfDistribution(), stat_sig.getStdOfDistribution(), stat_sig.pValue]
            else:
                pairwise[epo_i, 0] = result
            
        return pairwise
    
    def plot_it(self, it_matrix: np.ndarray):

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

            plt.matshow(it_matrix[:,:,epo_i,0], cmap='BuPu', vmin=0)

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
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)

            highest = np.max(it_matrix[:,:,epo_i,0])
            channel_pair_with_highest = np.unravel_index(np.argmax(it_matrix[:,:,epo_i,0]), it_matrix[:,:,epo_i,0].shape)
            print(f"Strongest regions: (Source Channel {self.channel_names[0][0][channel_pair_with_highest[0]]} --> " +
                                    f" Target Channel {self.channel_names[1][0][channel_pair_with_highest[1]]}) = {highest}")
            
            plt.show()
    

    def compute_mi(self):
        
        self.measure = 'Mutual Information'

        if self.estimator_type == 'histogram' or self.estimator_type == 'symbolic':
            self.mi_matrix = np.zeros((self.n_chan, self.n_chan, self.n_epo, 1)) # TEMPORARY, until I figure out how to get p-values for hist/bin and symbolic MI estimators
        
        else:
            self.mi_matrix = np.zeros((self.n_chan, self.n_chan, self.n_epo, 4))
        
        self.estimator = self.which_mi_estimator()

        for i in tqdm(range(self.n_chan)):
            for j in range(self.n_chan):

                if self.inter_brain or i != j:
                    s1, s2 = (self.eeg1[:, i, :], self.eeg2[:, j, :]) if self.is_epoched else (self.eeg1[i, :], self.eeg2[j, :])
                    
                    if self.estimator_type == 'histogram':
                        self.mi_matrix[i, j] = self.mi_hist(s1, s2)
                    elif self.estimator_type == 'symbolic':
                        self.mi_matrix[i, j] = self.mi_symb(s1, s2)
                    else:
                        self.mi_matrix[i, j] = self.estimate_it(s1, s2)

                    if not self.inter_brain:
                        self.mi_matrix[j, i] = self.mi_matrix[i, j]

        mi = np.array((self.mi_matrix))

        if self.vis:
            self.plot_it(mi)

        return mi
    
    def compute_te(self):

        self.measure = 'Transfer Entropy'
        self.estimator = self.which_te_estimator()

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
    # eeg_data1 = np.stack([A, B, C], axis = 1) #  10, 3, 100 (epo, ch, sample)
    # eeg_data2 = np.stack([D, E, F], axis = 1) #  10, 3, 100 (epo, ch, sample)

    ### FOR UNEPOCHED DATA
    #eeg_data1 = np.vstack([A, B, C]) #  3, 1000 (ch, sample)
    #eeg_data = np.vstack([D, E, F])  #  3, 1000 (ch, sample)


    eeg_data = np.vstack([A, B, C, D, E, F])
    print(eeg_data.shape)
    channel_names = [['A', 'B', 'C', 'D', 'E', 'F'], ['A', 'B', 'C', 'D', 'E', 'F']]

    it = HyperIT(eeg_data, eeg_data, channel_names=channel_names, calc_sigstats=False, estimator_type='symbolic', vis=True)
    it.compute_mi()
    #it.compute_te()
    
    #te_matrix_xy, te_matrix_yx = it.compute_te()