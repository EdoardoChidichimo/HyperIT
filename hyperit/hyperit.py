import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from PIL import Image, ImageDraw
from typing import Tuple, List, Union, Any
from tqdm import tqdm
import matplotlib.pyplot as plt

from jpype import isJVMStarted, startJVM, getDefaultJVMPath
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

from utils import setup_JArray, bandpass_filter_data, convert_names_to_indices, convert_indices_to_names, set_estimator, text_positions

from enum import Enum

class MeasureType(Enum):
    MI = 'mi' # Mutual Information
    TE = 'te' # Transfer Entropy
    PhyID = 'phyid' # Integrated Information Decomposition

    def __str__(self):
        return self.value  


class HyperIT:
    """ HyperIT: Hyperscanning Analyses using Information Theoretic Measures.

        HyperIT is equipped to compute pairwise, multivariate Mutual Information (MI), Transfer Entropy (TE), and Integrated Information Decomposition (ΦID) for continuous time-series data. 
        Compatible for both intra-brain and inter-brain analyses and for both epoched and unepoched data. 
        Multiple estimator choices and parameter customisations (via JIDT) are available, including KSG, Kernel, Gaussian, Symbolic, and Histogram/Binning. 
        Integrated statistical significance testing using permutation/boostrapping approach. 
        Visualisations of MI/TE matrices and information atoms/lattices also provided.

    Note: This class requires numpy, mne, matplotlib, PIL, jpype (with the local infodynamics.jar file), and phyid as dependencies.
    """

    ## SETTING UP JVM

    _jvm_initialised = False

    @classmethod
    def setup_JVM(cls, jarLocation: str, verbose: bool = False) -> None:
        """Setup JVM if not already started. To be called once before creating any instances."""
        if not cls._jvm_initialised:
            if not isJVMStarted():
                startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jarLocation}")
                cls._jvm_initialised = True
                if verbose:
                    print("JVM started successfully.")
            else:
                if verbose:
                    print("JVM already started.")
        else:
            if verbose:
                print("JVM setup already completed.")



    def __init__(self, data1: np.ndarray, data2: np.ndarray, channel_names: List[str], sfreq: float = None, freq_bands: dict = None, standardise_data: bool = False, inter_brain: bool = True, verbose: bool = False, **filter_options) -> None:
        """ Creates HyperIT object containing time-series data and channel names for analysis. 
            Automatic data checks for consistency and dimensionality, identifying whether analysis is to be intra- or inter-brain.

            Determines whether epochality of data.
                - If data is 3 dimensional, data is assumed to be epoched with shape    (epochs, channels, time_points).
                - If data is 2 dimensional, data is assumed to be unepoched with shape          (channels, time_points).

        Args:
            data1                   (np.ndarray): Time-series data for participant 1. Can take shape (n_epo, n_chan, n_samples) or (n_chan, n_samples) for epoched and unepoched data, respectively. 
            data2                   (np.ndarray): Time-series data for participant 1. Must have the same shape as data1.
            channel_names            (List[str]): A list of strings representing the channel names for each participant. [[channel_names_p1], [channel_names_p2]] or [[channel_names_p1]] for intra-brain.
            sfreq              (float, optional): Sampling frequency of the data.
            freq_bands          (dict, optional): Dictionary of frequency bands for bandpass filtering. {band_name: (low_freq, high_freq)}.
            standardise_data    (bool, optional): Whether to standardise the data before analysis. Defaults to True.
            inter_brain         (bool, optional): Whether the analysis is inter-brain (True) or intra-brain (False). Defaults to True.
            verbose             (bool, optional): Whether constructor and analyses should output details and progress. Defaults to False.
            **filter_options    (dict, optional): Additional keyword arguments for bandpass filtering.
        """

        if not self.__class__._jvm_initialised:
            raise RuntimeError("JVM has not been started. Call setup_JVM() before creating any instances of HyperIT.")

        self.verbose: bool = verbose

        self._channel_names = channel_names #  [[p1][p2]] or [[p1]] for intra-brain
        self._channel_indices1 = []
        self._channel_indices2 = []
        self._sfreq = sfreq if sfreq else None
        self._freq_bands = freq_bands if freq_bands else None
        self._filter_options = filter_options
        self._standardise_data = standardise_data
        self._inter_brain = inter_brain

        self._data1 = data1
        self._data2 = data2 

        self.__check_data()
        _, self._n_freq_bands, self._n_epo, self._n_chan, self._n_samples = self._all_data.shape

        self._roi = []
        self._roi_specified = False
        self._scale_of_organisation = 1 # 1 = micro organisation (single channel pairwise), n = meso- or n-scale organisation (n-sized groups of channels)
        self._initialise_parameter = None

        if self.verbose:
            print("HyperIT object created successfully.")

            if self._n_epo > 1:
                print(f"{'Inter-Brain' if self._inter_brain else 'Intra-Brain'} analysis and epoched data detected. \nAssuming data passed have shape ({self._n_epo} epochs, {self._n_chan} channels, {self._n_samples} time points).")
            else:
                print(f"{'Inter-Brain' if self._inter_brain else 'Intra-Brain'} analysis and unepoched data detected. \nAssuming data passed have shape ({self._n_chan} channels, {self._n_samples} time points).")

            if self._freq_bands:
                print(f"Data has been bandpass filtered: {self._freq_bands}.")


    ## DUNDER METHODS

    def __repr__(self) -> str:
        """ String representation of HyperIT object. """
        analysis_type = 'Hyperscanning' if self._inter_brain else 'Intra-Brain'
        channel_info = f"{self._channel_names[0]}"  # Assuming self._channel_names[0] is a list of channel names for the first data set
        
        # Adding second channel name if inter_brain analysis is being conducted
        if self._inter_brain:
            channel_info += f" and {self._channel_names[0][1]}"

        return (f"HyperIT Object: \n"
                f"{analysis_type} Analysis with {self._n_epo} epochs, {self._n_chan} channels, "
                f"and {self._n_samples} time points. \n"
                f"Channel names passed: \n"
                f"{channel_info}.")

    def __len__(self) -> int:
        """ Returns the number of epochs in the HyperIT object. """
        return self._all_data.shape
    
    def __str__(self) -> str:
        """ String representation of HyperIT object. """
        return self.__repr__()
    

    ## DATA, CHANNEL, & INITIALISATION CHECKING

    def __check_data(self) -> None:
        """ Checks the consistency and dimensionality of the time-series data and channel names. Sets the number of epochs, channels, and time points as object variables.

        Ensures:
            - Whether an intra-brain analysis is expected (when no second data set is provided or when data1==data2).
            - Data are numpy arrays.
            - Data shapes are consistent.
            - Data dimensions are either 2 or 3 dimensional.
            - Channel names are in correct format and match number of channels in data.
            - Data are bandpass filtered, if frequency bands are specified.
            - Data are standardised, if specified.
        """

        if self._data2 is None or np.array_equal(self._data1, self._data2) or self._data2.shape == (0,) or self._inter_brain == False:
            self._data2 = self._data1.copy()
            self._inter_brain = False
            
        if not all(isinstance(data, np.ndarray) for data in [self._data1, self._data2]):
            raise ValueError("Time-series data must be numpy arrays.")
        
        if self._data1.shape != self._data2.shape:
            raise ValueError("Time-series data must have the same shape for both participants.")
    
        if self._data1.ndim == 2:
            self._data1 = self._data1[np.newaxis, ...]
            self._data2 = self._data2[np.newaxis, ...]

        self._n_chan = self._data1.shape[1] 

        if self._data1.ndim not in [2,3]:
            raise ValueError(f"Unexpected number of dimensions in time-series data: {self._data1.ndim}. Expected 2 dimensions (channels, time_points) or 3 dimensions (epochs, channels, time_points).")

        if not isinstance(self._channel_names, list):
            raise ValueError("Channel names must be a list of strings or a list of lists of strings.")

        # Checking the type of the first element to determine the structure
        if isinstance(self._channel_names[0], str):
            # If the first element is a string, assume all are strings and check accordingly
            if not all(isinstance(name, str) for name in self._channel_names):
                raise ValueError("All elements must be strings if the first element is a string.")
            # Duplicate the list if it is intended for use in an intra-brain or similar context
            self._channel_names = [self._channel_names, self._channel_names.copy()]

        elif isinstance(self._channel_names[0], list):
            # If the first element is a list, ensure all elements are lists of strings
            if not all(isinstance(sublist, list) for sublist in self._channel_names):
                raise ValueError("All elements must be lists of strings if the first element is a list.")
            for sublist in self._channel_names:
                if not all(isinstance(name, str) for name in sublist):
                    raise ValueError("All sublists must be lists of strings.")

            # Here, check if there's only one sublist and duplicate if necessary
            if len(self._channel_names) == 1:
                self._channel_names = self._channel_names * 2

        else:
            raise ValueError("Channel names must be either a list of strings or a list of lists of strings.")

        if any(len(names) != self._n_chan for names in self._channel_names):
            raise ValueError("The number of channels in time-series data does not match the length of channel_names.")
        
        
        self._channel_indices1 = np.arange(len(self._channel_names[0]))
        self._channel_indices2 = np.arange(len(self._channel_names[1])) if len(self._channel_names) > 1 else self._channel_indices2.copy()


        if self._freq_bands:
            self._data1, self._data2 = bandpass_filter_data(self._data1, self._sfreq, self._freq_bands, **self._filter_options), bandpass_filter_data(self._data2, self._sfreq, self._freq_bands, **self._filter_options)
                
        else:
            self._data1, self._data2 = np.expand_dims(self._data1, axis=0), np.expand_dims(self._data2, axis=0)

        if self._standardise_data:
            self._data1 = (self._data1 - np.mean(self._data1, axis=-1, keepdims=True)) / np.std(self._data1, axis=-1, keepdims=True)
            self._data2 = (self._data2 - np.mean(self._data2, axis=-1, keepdims=True)) / np.std(self._data2, axis=-1, keepdims=True)
            
        self._all_data = np.stack([self._data1, self._data2], axis=0)

    
    
    ## DEFINING REGIONS OF INTEREST

    @property
    def roi(self) -> List[List[Union[str, int, list]]]:
        """Regions of interest for both data of the HyperIT object (defining spatial scale of organisation). To set this, call .roi(roi_list). 
        
        HyperIT is defaulted to **micro-scale** analysis (individual channels) but specific channels can be specified for pointwise comparison: ``roi_list = [['Fp1', 'Fp2'], ['F3', 'F4']]``, for example. 
        For **meso-scale** analysis (clusters of channels), equally-sized and equally-numbered clusters must be defined for both sets of data in the following way: ``roi_list = [[[PP1_cluster_1], ..., [PP1_cluster_n]], [[PP2_cluster_1], ..., [PP2_cluster_n]]]``. 
        Finally, for **macro-scale** analysis (all channels per person), the specification can be set as ``roi_list = [[PP1_all_channels][PP2_all_channels]]`` (note that PP1_all_channels and PP2_all_channels should be list themselves).
        Importantly, as long as the ``channel_names`` are instantiated properly in the initiation of the HyperIT object, the ROI can even be given as a lists of channel indices (integers). 
        In any case, to set these scales of organisations, simply amend the ``roi`` property of the HyperIT object used.
        Call ``roi.reset_roi()`` to reset the ROI to all channels.
        
        """
        return self._roi

    @roi.setter
    def roi(self, roi_list: List[List[Union[str, int, list]]]) -> None:
        """Sets the region of interest for both data of the HyperIT object.

        Args:
            roi_list: A list of lists, where each sublist is a ROI containing either strings of EEG channel names or integer indices or multiple ROIs formed as another list.
        
        Raises:
            ValueError: If the value is not a list of lists, if elements of the sublists are not of type str or int, or if sublists do not have the same length.
        """

        self._roi_specified = True

        ## DETERMINE SCALE OF ORGANISATION
        # 1: Micro organisation (specified channels, pairwise comparison)           e.g., roi_list = [['Fp1', 'Fp2'], ['F3', 'F4']]

        # n: Meso- or n-scale organisation (n specified channels per ROI group)     e.g., roi_list = [[  ['Fp1', 'Fp2'], ['CP1', 'CP2']   ],   n CHANNELS IN EACH GROUP FOR PARTICIPANT 1
                                                                                                    # [    ['F3', 'F4'], ['F7', 'F8']     ]]   n CHANNELS IN EACH GROUP FOR PARTICIPANT 2

        # Check if roi_list is structured for pointwise channel comparison
        if all(isinstance(sublist, list) and not any(isinstance(item, list) for item in sublist) for sublist in roi_list):
            self._scale_of_organisation = 1 

        # Check if roi_list is structured for grouped channel comparison
        elif all(isinstance(sublist, list) and all(isinstance(item, list) for item in sublist) for sublist in roi_list):
            # Ensure uniformity in the number of groups across both halves
            num_groups_x = len(roi_list[0])
            num_groups_y = len(roi_list[1])
            if num_groups_x == num_groups_y:
                self._soi_groups = num_groups_x 

                group_lengths = [len(group) for half in roi_list for group in half]
                if len(set(group_lengths)) == 1:
                    self._scale_of_organisation = group_lengths[0] 
                    self._initialise_parameter = (self._scale_of_organisation, self._scale_of_organisation)
                else:
                    raise ValueError("Not all groups have the same number of channels.")
            else:
                raise ValueError("ROI halves do not have the same number of channel groups per participant.")

        else:
            raise ValueError("ROI structure is not recognised.")
        
        if self.verbose:
            print(f"Scale of organisation: {self._scale_of_organisation} channels.")
            print(f"Groups of channels: {self._soi_groups}")

        roi1, roi2 = roi_list
        
        self._channel_indices1, self._channel_indices2 = [convert_names_to_indices(self._channel_names, part, idx) for idx, part in enumerate(roi_list)] 
       
        # POINTWISE CHANNEL COMPARISON
        if self._scale_of_organisation == 1:
            self._data1 = self._data1[:, :, self._channel_indices1, :]
            self._data2 = self._data2[:, :, self._channel_indices2, :]
            
            self._n_chan = len(self._channel_indices1)

        # for other scales of organisation, this will be handled in the compute_mi and compute_te functions

        self._roi = [self._channel_indices1, self._channel_indices2]
        
    def reset_roi(self) -> None:
        """Resets the region of interest for both data of the HyperIT object to all channels."""
        self._roi_specified = False
        self._scale_of_organisation = 1
        self._channel_indices1 = np.arange(len(self._channel_names[0]))
        self._channel_indices2 = np.arange(len(self._channel_names[1]) if len(self._channel_names) > 1 else len(self._channel_names[0]))
        self._roi = [self._channel_indices1, self._channel_indices2]
        self._data1, self._data2 = self._all_data
        self._n_chan = len(self._channel_indices1)
        print("Region of interest has been reset to all channels.")



    ## HARD-CODED HISTOGRAM AND SYMBOLIC MI ESTIMATION FUNCTIONS

    def __estimate_mi_hist(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Calculates Mutual Information using Histogram/Binning Estimator for time-series signals."""

        def calc_fd_bins(X: np.ndarray, Y: np.ndarray) -> int:
            """Calculates the optimal frequency-distribution bin size for histogram estimator using Freedman-Diaconis Rule."""

            fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
            fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
            fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
            return fd_bins
        
        def hist_calc_mi(X, Y):
            # Joint probability distribution
            j_hist, _, _ = np.histogram2d(X, Y, bins=calc_fd_bins(X, Y))
            pxy = j_hist / np.sum(j_hist)  # Joint probability distribution

            # Marginals probability distribution
            px = np.sum(pxy, axis=1) 
            py = np.sum(pxy, axis=0) 

            # Entropies
            Hx = -np.sum(px * np.log2(px + np.finfo(float).eps))
            Hy = -np.sum(py * np.log2(py + np.finfo(float).eps))
            Hxy = -np.sum(pxy * np.log2(pxy + np.finfo(float).eps))

            return Hx + Hy - Hxy

        result = hist_calc_mi(s1, s2)

        if self.calc_sigstats:
            permuted_mi_values = []

            for _ in range(self.stat_sig_perm_num):
                np.random.shuffle(s2)
                permuted_mi = hist_calc_mi(s1, s2)
                permuted_mi_values.append(permuted_mi)

            mean_permuted_mi = np.mean(permuted_mi_values)
            std_permuted_mi = np.std(permuted_mi_values)
            p_value = np.sum(permuted_mi_values >= result) / self.stat_sig_perm_num
            return np.array([result, mean_permuted_mi, std_permuted_mi, p_value])

        return np.array([result])

    def __estimate_mi_symb(self, s1: np.ndarray, s2: np.ndarray, l: int = 1, k: int = 3) -> float:
        """Calculates Mutual Information using Symbolic Estimator for time-series signals.

            l: time delay or lag (i.e., how many time points to skip)
            k: embedding dimension (i.e., how many time points to consider in each symbol)
        
        """

        symbol_weights = np.power(k, np.arange(k))

        def symb_symbolise(X: np.ndarray, l: int, k: int) -> np.ndarray:
            Y = np.empty((k, len(X) - (k - 1) * l))
            for i in range(k):
                Y[i] = X[i * l:i * l + Y.shape[1]]
            return Y.T

        def symb_normalise_counts(d) -> None:
            total = sum(d.values())        
            return {key: value / total for key, value in d.items()}
        
        def symb_calc_mi(X, Y, l, k):
            X = symb_symbolise(X, l, k).argsort(kind='quicksort')
            Y = symb_symbolise(Y, l, k).argsort(kind='quicksort')

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

            return entropy_X + entropy_Y - entropy_XY


        result = symb_calc_mi(s1, s2, l, k)

        if self.calc_sigstats:
            permuted_mi_values = []

            for _ in range(self.stat_sig_perm_num):
                np.random.shuffle(s2)
                permuted_mi = symb_calc_mi(s1, s2, l, k)
                permuted_mi_values.append(permuted_mi)

            mean_permuted_mi = np.mean(permuted_mi_values)
            std_permuted_mi = np.std(permuted_mi_values)
            p_value = np.sum(permuted_mi_values >= result) / self.stat_sig_perm_num
            return np.array([result, mean_permuted_mi, std_permuted_mi, p_value])

        return np.array([result])



    ## ESTIMATION AND HELPER FUNCTIONS

    def __which_estimator(self, measure: str) -> None:
        """Determines the estimator to use based on the measure type and sets the estimator, properties, and initialisation parameters."""

        self.estimator_name, calculator, properties, initialise_parameter = set_estimator(self.estimator_type, measure, self.params) # from utils.py function

        if calculator:
            self.Calc = calculator()

        if properties:
            for key, value in properties.items():
                self.Calc.setProperty(key, value)

        if initialise_parameter:
            self._initialise_parameter = initialise_parameter

    def __setup_matrices(self) -> int:
        """ Sets up the matrices for Mutual Information, Transfer Entropy, or Integrated Information Decomposition. """

        n_combo = self._n_chan if self._scale_of_organisation == 1 else self._soi_groups

        if self.measure == MeasureType.MI or self.measure == MeasureType.TE:
            self.it_matrix_xy = np.zeros((self._n_freq_bands, self._n_epo, n_combo, n_combo, 4))
            
            if self.measure == MeasureType.TE and self._inter_brain:
                # TE is assymetrical so will need to compute both X->Y and Y->X.
                # BUT, only need another matrix for interbrain as intra-brain channels are already all cycled through in first matrix (half 1 = X->Y, half 2 = Y->X)
                # MI is symmetrical and therefore all information is stored in one matrix
                self.it_matrix_yx = self.it_matrix_xy.copy()

        elif self.measure == MeasureType.PhyID:
            self.it_matrix_xy = [[[[{} for _ in range(n_combo)] for _ in range(n_combo)] for _ in range(self._n_epo)] for _ in range(self._n_freq_bands)]
            if self._inter_brain:
                self.it_matrix_yx = self.it_matrix_xy.copy() 

        return n_combo
        
    def __estimate_it(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """ Estimates Mutual Information or Transfer Entropy for a pair of time-series signals using JIDT estimators. """

        # Initialise parameter describes the dimensions of the data
        self.Calc.initialise(*self._initialise_parameter) if self._initialise_parameter else self.Calc.initialise()
        self.Calc.setObservations(setup_JArray(s1), setup_JArray(s2))
        result = self.Calc.computeAverageLocalOfObservations() * np.log(2)

        # Conduct significance testing
        if self.calc_sigstats:
            stat_sig = self.Calc.computeSignificance(self.stat_sig_perm_num)
            return np.array([result, stat_sig.getMeanOfDistribution(), stat_sig.getStdOfDistribution(), stat_sig.pValue])
            
        return np.array([result])
    
    def __estimate_atoms(self, s1: np.ndarray, s2: np.ndarray) -> None:

        atoms_results_xy, _ = calc_PhiID(s1, s2, tau=self.tau, kind='gaussian', redundancy=self.redundancy)

        if not atoms_results_xy:  
            raise ValueError("Empty results from calc_PhiID, critical data processing cannot continue.")

        calc_atoms_xy = np.mean(np.array([atoms_results_xy[_] for _ in PhiID_atoms_abbr]), axis=1)
        return {key: value for key, value in zip(atoms_results_xy.keys(), calc_atoms_xy)}

    def __filter_estimation(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """ Filters the estimation in case incompatible with JIDT. """

        if self.measure == MeasureType.MI:
            match self.estimator_type:
                case 'histogram':
                    return self.__estimate_mi_hist(s1, s2)
                case 'symbolic':
                    return self.__estimate_mi_symb(s1, s2)

        elif self.measure == MeasureType.PhyID:
            return self.__estimate_atoms(s1, s2)



        return self.__estimate_it(s1, s2)

    def __compute_pair_or_group(self, freq_band: int, epoch: int, i: int, j: int) -> None:
        """ Computes the Mutual Information or Transfer Entropy for a pair of channels or groups of channels. """

        if not (self._inter_brain or i != j):
            return

        channel_or_group_i = i if self._scale_of_organisation == 1 else self._roi[0][i]
        channel_or_group_j = j if self._scale_of_organisation == 1 else self._roi[1][j]    
        # data needs to be (samples, channels/groups) if not pointwise comparison 
        # (this is how both JIDT and phyid handle and expect incoming multivariate data)
        # (Note that .T does not affect pointwise comparison as it is already in the correct shape)
        s1, s2 = self._data1[freq_band, epoch, channel_or_group_i, :].T, self._data2[freq_band, epoch, channel_or_group_j, :].T

        if self.measure == MeasureType.PhyID:
            self.it_matrix_xy[freq_band][epoch][i][j] = self.__filter_estimation(s1, s2)
            if self._inter_brain:
                self.it_matrix_yx[freq_band][epoch][i][j] = self.__filter_estimation(s2, s1)
            return
        

        self.it_matrix_xy[freq_band, epoch, i, j] = self.__filter_estimation(s1, s2)

        if self.measure == MeasureType.MI and not self._inter_brain:
            self.it_matrix_xy[freq_band, epoch, j, i] = self.it_matrix_xy[freq_band, epoch, i, j]

        if self.measure == MeasureType.TE and self._inter_brain:
            self.it_matrix_yx[freq_band, epoch, i, j] = self.__filter_estimation(s2, s1)



    ## MAIN CALCULATION FUNCTIONS         

    def __main_calc(self) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:

        # Set up it_matrix_xy (and it_matrix_yx if necessary) and store the number of channels/groups to loop through
        loop_range = self.__setup_matrices()

        for freq_band in range(self._n_freq_bands):
            for epoch in range(self._n_epo):
                for i in tqdm(range(loop_range)):
                    for j in range(loop_range):
                        self.__compute_pair_or_group(freq_band, epoch, i, j)

        if self.vis:
            self.__prepare_vis()
            
        if not self.measure == MeasureType.MI and self._inter_brain:
            return self.it_matrix_xy, self.it_matrix_yx
        
        return self.it_matrix_xy

    def __setup_mite_calc(self, estimator_type: str, calc_sigstats: bool, vis: bool, **kwargs) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """ General function for computing Mutual Information or Transfer Entropy. """

        self.estimator_type = estimator_type.lower() if not (self.measure == MeasureType.MI and estimator_type.lower() == 'ksg') else 'ksg1'
        self.calc_sigstats = calc_sigstats
        self.vis = vis
        self.params = kwargs
        self.stat_sig_perm_num = self.params.get('stat_sig_perm_num', 100)
        self.p_threshold = self.params.get('p_threshold', 0.05)
        
        # Set up the estimator and properties
        self.__which_estimator(str(self.measure))
        
        return self.__main_calc()

    def __setup_atom_calc(self, tau: int, redundancy: str, vis: bool, **kwargs) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """ General function for computing Integrated Information Decomposition. """

        self.tau = tau
        self.redundancy = redundancy
        self.vis = vis
        self.params = kwargs

        return self.__main_calc()



    ## VISUALISATION FUNCTIONS

    def __prepare_vis(self) -> None:
        """ Prepares the visualisation of Mutual Information, Transfer Entropy, or Integrated Information Decomposition matrix/matrices. """

        if self.verbose:
            print(f"Plotting {self.measure_title}...")

        if self.measure == MeasureType.PhyID:
            self.__plot_atoms(self.it_matrix_xy)
            if self._inter_brain:
                self.__plot_atoms(self.it_matrix_yx)
            return
    
        self.__plot_it(self.it_matrix_xy)

        if self.measure == MeasureType.TE and self._inter_brain:
            self.__plot_it(self.it_matrix_yx)

    def __plot_it(self, it_matrix: np.ndarray) -> None:
        """Plots the Mutual Information or Transfer Entropy matrix for visualisation. 
        Axes labelled with source and target channel names. 
        Choice to plot for all epochs, specific epoch(s), or average across epochs.

        Args:
            it_matrix (np.ndarray): The Mutual Information or Transfer Entropy matrix to be plotted with shape (n_chan, n_chan, n_epo, 4), 
            where the last dimension represents the statistical signficance testing results: (local result, distribution mean, distribution standard deviation, p-value).
        """

        title = f'{self.measure_title} | {self.estimator_name} \n {"Inter-Brain" if self._inter_brain else "Intra-Brain"}'
        epochs = [0] # default to un-epoched or epoch-average case
        choice = 3 # default to average MI/TE across epochs
        
        if self._scale_of_organisation > 1:
            source_channel_names = convert_indices_to_names(self._channel_names, self._channel_indices1, 0) if self._scale_of_organisation == 1 else convert_indices_to_names(self._channel_names, self._roi[0], 0)
            target_channel_names = convert_indices_to_names(self._channel_names, self._channel_indices2, 1) if self._scale_of_organisation == 1 else convert_indices_to_names(self._channel_names, self._roi[1], 1)

            print("Plotting for grouped channels.")

            print("Source Groups:")
            for i in range(self._soi_groups):
                print(f"{i+1}: {source_channel_names[i]}")

            print("\nTarget Groups:")
            for i in range(self._soi_groups):
                print(f"{i+1}: {target_channel_names[i]}")


        if self._n_epo > 1: 
            choice = input(f"{self._n_epo} epochs detected. Plot for \n1. All epochs \n2. Specific epoch \n3. Average MI/TE across epochs \nEnter choice: ")
            if choice == "1":
                print("Plotting for all epochs.")
                epochs = range(self._n_epo)
            elif choice == "2":
                epo_choice = input(f"Enter epoch number(s) [1 to {self._n_epo}] separated by comma only: ")
                try:
                    epochs = [int(epo)-1 for epo in epo_choice.split(',')]
                except ValueError:
                    print("Invalid input. Defaulting to plotting all epochs.")
                    epochs = range(self._n_epo)
            elif choice == "3":
                print("Plotting for average MI/TE across epochs. Note that p-values will not be shown.")
                
            else:
                print("Invalid choice. Defaulting to un-epoched data.")

        for freq_band in range(self._n_freq_bands):
            for epo_i in epochs:
                
                highest = np.max(it_matrix[freq_band, epo_i, :, :, 0])
                channel_pair_with_highest = np.unravel_index(np.argmax(it_matrix[freq_band, epo_i, :, :, 0]), it_matrix[freq_band, epo_i, :, :, 0].shape)
                if self.verbose:
                    if self._scale_of_organisation == 1:
                        print(f"Strongest regions: (Source Channel {convert_indices_to_names(self._channel_names, self._channel_indices1, 0)[channel_pair_with_highest[0]]} --> " +
                                            f" Target Channel {convert_indices_to_names(self._channel_names, self._channel_indices2, 1)[channel_pair_with_highest[1]]}) = {highest}")
                    else:
                        print(f"Strongest regions: (Source Group {source_channel_names[i]} --> Target Group {target_channel_names[i]}) = {highest}")

                plt.figure(figsize=(12, 10))
                plt.imshow(it_matrix[freq_band, epo_i, :, :, 0], cmap='BuPu', vmin=0, aspect='auto')

                band_description = ""
                if self._freq_bands:
                    band_name, band_range = list(self._freq_bands.items())[freq_band]
                    band_description = f"{band_name}: {band_range}"

                # Check if epochs are involved and adjust the title accordingly
                if self._n_epo > 1 and not choice == "3":
                    if band_description:
                        plt.title(f'{title}; Frequency Band {band_description}, Epoch {epo_i+1}', pad=20)
                    else:
                        plt.title(f'{title}; Epoch {epo_i+1}', pad=20)
                else:
                    if band_description:
                        plt.title(f'{title}; Frequency Band {band_description}', pad=20)
                    else:
                        plt.title(title, pad=20)
                    
                if self.calc_sigstats and not choice == "3":
                    for i in range(it_matrix.shape[0]):
                        for j in range(it_matrix.shape[1]):
                            p_val = float(it_matrix[freq_band, epo_i, i, j, 3])
                            if p_val < self.p_threshold and (not self._inter_brain and i != j):
                                normalized_value = (it_matrix[freq_band, epo_i, i, j, 0] - np.min(it_matrix[freq_band, epo_i, :, :, 0])) / (np.max(it_matrix[freq_band, epo_i, :, :, 0]) - np.min(it_matrix[freq_band, epo_i, :, :, 0]))
                                text_colour = 'white' if normalized_value > 0.5 else 'black'
                                plt.text(j, i, f'p={p_val:.2f}', ha='center', va='center', color=text_colour, fontsize=8, fontweight='bold')

                plt.colorbar()
                plt.xlabel('Target')
                plt.ylabel('Source')

                if self._scale_of_organisation == 1:
                    plt.xticks(range(self._n_chan), convert_indices_to_names(self._channel_names, self._channel_indices2, 1), rotation=90) 
                    plt.yticks(range(self._n_chan), convert_indices_to_names(self._channel_names, self._channel_indices1, 0))
                else:
                    plt.xticks(range(self._soi_groups), [f'Group {i+1}' for i in range(self._soi_groups)], rotation=90)
                    plt.yticks(range(self._soi_groups), [f'Group {i+1}' for i in range(self._soi_groups)])

                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=True)
                plt.tick_params(axis='y', which='both', right=False, left=True, labelleft=True)
                plt.show()

    def __plot_atoms(self, phi_dict: dict) -> None:
        """Plots the values of the atoms in the lattice for each frequency band for a given pair of channels/ROI groups."""
        
        if not phi_dict or not phi_dict[0]:
            print("The phi_dict is empty or not properly structured.")
            return

        f_range = len(phi_dict)
        x_range = len(phi_dict[0])
        y_range = len(phi_dict[0][0])
        print("f_range:", f_range)

        for freq_band in range(f_range):
            print(f"Now plotting for frequency band {freq_band}")
            prompt_message = f"Choose two numbers from this range for X and Y channels/groups: [0-{x_range - 1}], [0-{y_range - 1}] (or type 'done' to stop): "
            
            while True:
                user_input = input(prompt_message).split(',')
                
                if len(user_input) == 1 and user_input[0].lower() == 'done':
                    break

                if len(user_input) != 2 or not all(part.strip().isdigit() for part in user_input):
                    print("Invalid input. Please enter exactly two numbers separated by a comma.")
                    continue

                ch_X, ch_Y = (int(part.strip()) for part in user_input)

                try:
                    value_dict = phi_dict[freq_band][ch_X][ch_Y]
                    if value_dict is None:
                        raise KeyError

                    image = Image.open('visualisations/atoms_lattice_values.png')
                    draw = ImageDraw.Draw(image) 

                    for text, pos in text_positions.items():
                        value = value_dict.get(text, '0')
                        plot_text = f"{round(float(value), 3):.3f}"
                        draw.text(pos, plot_text, fill="black", font_size=25)

                    image.show()
                
                except KeyError:
                    print("Invalid channel/group indices.")



    ## HIGH-LEVEL INTERFACE FUNCTIONS

    def compute_mi(self, estimator_type: str = 'kernel', calc_sigstats: bool = False, vis: bool = False, **kwargs) -> np.ndarray:
        """
        Function to compute mutual information between data (time-series signals) instantiated in the HyperIT object.

        PARAMETER OPTIONS FOR MUTUAL INFORMATION ESTIMATORS (defaults in parentheses):
        Estimator types:    kwargs:
        - histogram:        None
        - ksg1:             kraskov_param (4), normalise (True)
        - ksg2:             kraskov_param (4), normalise (True)
        - kernel:           kernel_width (0.25), normalise (True)
        - gaussian:         normalise (True)
        - symbolic:         l (1), m (3)

        Args:
            estimator_type       (str, optional): Which mutual information estimator to use. Defaults to 'kernel'.
            calc_sigstats       (bool, optional): Whether to conduct statistical signficance testing. Defaults to False.
            vis                 (bool, optional): Whether to visualise (via __plot_it()). Defaults to False.

        Returns:
                                    (np.ndarray): A matrix of mutual information values with shape (n_chan, n_chan, n_epo, 4),
                                                  where the last dimension represents the statistical signficance testing results: (local result, distribution mean, distribution standard deviation, p-value). 
                                                  If calc_sigstats is False, only the local results will be returned in this last dimension.
        """
        
        self.measure = MeasureType.MI
        self.measure_title = 'Mutual Information'
        return self.__setup_mite_calc(estimator_type, calc_sigstats, vis, **kwargs)

    def compute_te(self, estimator_type: str = 'kernel', calc_sigstats: bool = False, vis: bool = False, **kwargs) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Function to compute transfer entropy between data (time-series signals) instantiated in the HyperIT object. 
            data1 is taken to be the source and data2 the target (X -> Y). This function automatically computes the opposite matrix, too (Y -> X).

        PARAMETER OPTIONS FOR TRANSFER ENTROPY ESTIMATORS (defaults in parentheses):
        Estimator types:    kwargs:
        - ksg:              k, k_tau, l, l_tau, delay, kraskov_param (all 1), normalise (True)
        - kernel:           kernel_width (0.5), normalise (True)
        - gaussian:         k, k_tau, l, l_tau, delay (all 1), bias_correction (False), normalise (True)
        - symbolic:         k (1), normalise (True)

        Args:
            estimator_type       (str, optional): Which transfer entropy estimator to use. Defaults to 'kernel'.
            calc_sigstats       (bool, optional): Whether to conduct statistical signficance testing. Defaults to False.
            vis                 (bool, optional): Whether to visualise (via __plot_it()). Defaults to False.

        Returns:
                   Tuple(np.ndarray, np.ndarray): Two matrices of transfer entropy values (X->Y and Y->X), each with shape (n_chan, n_chan, n_epo, 4),
                                                  where the last dimension represents the statistical signficance testing results: (local result, distribution mean, distribution standard deviation, p-value). 
                                                  If calc_sigstats is False, only the local results will be returned in this last dimension.
        """
        
        self.measure = MeasureType.TE
        self.measure_title = 'Transfer Entropy'
        return self.__setup_mite_calc(estimator_type, calc_sigstats, vis, **kwargs)

    def compute_atoms(self, tau: int = 1, redundancy: str = 'MMI', vis: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to compute Integrated Information Decomposition (ΦID) between data (time-series signals) instantiated in the HyperIT object.
            Option to visualise the lattice values for a specific channel pair (be sure to specify via plot_channels kwarg).

        Args:
            tau             (int, optional): Time-lag parameter. Defaults to 1.
            kind            (str, optional): Estimator type. Defaults to "gaussian".
            redundancy      (str, optional): Redundancy function to use. Defaults to 'MMI' (Minimum Mutual Information).
            vis            (bool, optional): Whether to visualise (via __plot_atoms()). Defaults to False.

        Returns:
              Tuple(np.ndarray, np.ndarray): Two matrices of Integrated Information Decomposition dictionaries (representing all atoms, both X->Y and Y->X), each with shape (n_chan, n_chan),
        """
        
        self.measure = MeasureType.PhyID
        self.measure_title = 'Integrated Information Decomposition'
        return self.__setup_atom_calc(tau, redundancy, vis, **kwargs)