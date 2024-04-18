import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from PIL import Image, ImageDraw
from typing import Tuple, List, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

from jpype import JPackage, shutdownJVM, isJVMStarted, JPackage, startJVM, getDefaultJVMPath
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

from utils import setup_JArray, bandpass_filter_data, convert_names_to_indices, convert_indices_to_names, set_estimator, text_positions


class HyperIT:
    """ HyperIT: Hyperscanning Analyses using Information Theoretic Measures.

        HyperIT is equipped to compute pairwise, multivariate Mutual Information (MI), Transfer Entropy (TE), and Integrated Information Decomposition (ΦID) for continuous time-series data. 
        Compatible for both intra-brain and inter-brain analyses and for both epoched and unepoched data. 
        Multiple estimator choices and parameter customisations (via JIDT) are available, including KSG, Kernel, Gaussian, Symbolic, and Histogram/Binning. 
        Integrated statistical significance testing using permutation/boostrapping approach. 
        Visualisations of MI/TE matrices and information atoms/lattices also provided.

    Args:
        ABC (_type_): Abstract Base Class for HyperIT.

    Note: This class requires numpy, mne, matplotlib, PIL, jpype (with the local infodynamics.jar file), and phyid as dependencies.
    """

    _jvm_initialised = False

    @classmethod
    def setup_JVM(cls, working_directory: str, verbose: bool = True) -> None:
        """Setup JVM if not already started. To be called once before creating any instances."""
        if not cls._jvm_initialized:
            if not isJVMStarted():
                startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_location}")
                cls._jvm_initialized = True
                if verbose:
                    print("JVM started successfully.")
            else:
                if verbose:
                    print("JVM already started.")
        else:
            if verbose:
                print("JVM setup already completed.")

    def __init__(self, data1: np.ndarray, data2: np.ndarray, channel_names: List[str], sfreq: float, freq_bands: dict, standardise_data: bool = False, verbose: bool = False, working_directory: str = None, **filter_options):
        """ Creates HyperIT object containing time-series data and channel names for analysis. 
            Automatic data checks for consistency and dimensionality, identifying whether analysis is to be intra- or inter-brain.

            Determines whether epochality of data.
                - If data is 3 dimensional, data is assumed to be epoched with shape    (epochs, channels, time_points).
                - If data is 2 dimensional, data is assumed to be unepoched with shape          (channels, time_points).

        Args:
            data1                   (np.ndarray): Time-series data for participant 1. Can take shape (n_epo, n_chan, n_samples) or (n_chan, n_samples) for epoched and unepoched data, respectively. 
            data2                   (np.ndarray): Time-series data for participant 1. Must have the same shape as data1.
            channel_names            (List[str]): A list of strings representing the channel names for each participant. [[channel_names_p1], [channel_names_p2]] or [[channel_names_p1]] for intra-brain.
            sfreq                        (float): Sampling frequency of the data.
            freq_bands                    (dict): Dictionary of frequency bands for bandpass filtering. {band_name: (low_freq, high_freq)}.
            standardise_data    (bool, optional): Whether to standardise the data before analysis. Defaults to True.
            verbose             (bool, optional): Whether constructor and analyses should output details and progress. Defaults to False.
            working_directory    (str, optional): The directory where the infodynamics.jar file is located. Defaults to None (later defaults to os.getcwd()).
        """

        if not self.__class__._jvm_initialised:
            raise RuntimeError("JVM has not been started. Call setup_JVM() before creating any instances of HyperIT.")

        self.verbose: bool = verbose

        setup_JVM(working_directory, self.verbose)

        self._channel_names = channel_names #  [[p1][p2]] or [[p1]] for intra-brain
        self._channel_indices1 = []
        self._channel_indices2 = []
        self._sfreq = sfreq if sfreq else None
        self._freq_bands = freq_bands if freq_bands else None
        self._filter_options = filter_options

        self._all_data = [data1, data2]
        self._data1, self._data2 = self._all_data
        self._standardise_data = standardise_data

        # NOTE: _data1, _data2, _channel_indices1, _channel_indices2 will be used for calculations (as these can be amended during ROI setting)

        self._roi = []
        self._roi_specified = False
        self._scale_of_organisation = 1 # 0 = global organisation (all channels), 1 = micro organisation (each channel), n = meso- or n-scale organisation (n channels)
        
        self._inter_brain: bool = not np.array_equal(data1, data2)
        self._initialise_parameter = None

        self.__check_data()


        _, self._n_freq_bands, self._n_epo, self._n_chan, self._n_samples = self._all_data.shape
        print("Data shape: ", self._all_data.shape)

        if self.verbose:
            print("HyperIT object created successfully.")

            if self._n_epo > 1:
                print(f"{'Inter-Brain' if self._inter_brain else 'Intra-Brain'} analysis and epoched data detected. \nAssuming data passed have shape ({self._n_epo} epochs, {self._n_chan} channels, {self._n_samples} time points).")
            else:
                print(f"{'Inter-Brain' if self._inter_brain else 'Intra-Brain'} analysis and unepoched data detected. \nAssuming data passed have shape ({self._n_chan} channels, {self._n_samples} time points).")

            if self._freq_bands:
                print(f"Data has been bandpass filtered: {self._freq_bands}.")


    def __del__(self):
        """ Destructor for HyperIT object. Ensures JVM is shutdown upon deletion of object. """
        try:
            if isJVMStarted():
                shutdownJVM()
                if self.verbose:
                    print("JVM has been shutdown.")
        except Exception as e:
            if self.verbose:
                print(f"Error shutting down JVM: {e}")

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
    

    def __check_data(self) -> None:
        """ Checks the consistency and dimensionality of the time-series data and channel names. Sets the number of epochs, channels, and time points as object variables.

        Ensures:
            - Data are numpy arrays.
            - Data shapes are consistent.
            - Data dimensions are either 2 or 3 dimensional.
            - Channel names are in correct format and match number of channels in data.
            - Data are bandpass filtered, if frequency bands are specified.
            - Data are standardised, if specified.
        """

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

        if not isinstance(self._channel_names, (list, np.ndarray)) or isinstance(self._channel_names[0], str):
            raise ValueError("Channel names must be a list of strings or a list of lists of strings for inter-brain analysis.")
    
        if not self._inter_brain and isinstance(self._channel_names[0], list):
            self._channel_names = [self._channel_names] * 2

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
        
        self._channel_indices1 = convert_names_to_indices(self._channel_names, roi1, 0) # same array as roi1 just with indices instead of EEG channel names
        self._channel_indices2 = convert_names_to_indices(self._channel_names, roi2, 1)

        # POINTWISE CHANNEL COMPARISON
        if self._scale_of_organisation == 1:
            self._data1 = self._data1[:, self._channel_indices1, :]
            self._data2 = self._data2[:, self._channel_indices2, :]
            
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



    def __mi_hist(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Calculates Mutual Information using Histogram/Binning Estimator for time-series signals."""

        def calc_fd_bins(X: np.ndarray, Y: np.ndarray) -> int:
            """Calculates the optimal frequency-distribution bin size for histogram estimator using Freedman-Diaconis Rule."""

            fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
            fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
            fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
            return fd_bins
        
        def hist_calc_mi(X, Y):
            j_hist, _, _ = np.histogram2d(X, Y, bins=calc_fd_bins(X, Y))
            pxy = j_hist / np.sum(j_hist)  # Joint probability distribution

            # Marginals
            px = np.sum(pxy, axis=1) 
            py = np.sum(pxy, axis=0) 

            # Entropies
            Hx = -np.sum(px * np.log2(px + np.finfo(float).eps))
            Hy = -np.sum(py * np.log2(py + np.finfo(float).eps))
            Hxy = -np.sum(pxy * np.log2(pxy + np.finfo(float).eps))

            return Hx + Hy - Hxy

        estimations = np.zeros((self._n_epo, 4)) # stores MI result, mean, std, p-value per epoch

        for epo_i in range(self._n_epo):

            x, y = (s1[epo_i, :], s2[epo_i, :]) 
            mi = hist_calc_mi(x, y)

            if self.calc_sigstats:
                permuted_mi_values = []

                for _ in range(self.stat_sig_perm_num):
                    np.random.shuffle(y)
                    permuted_mi = hist_calc_mi(x, y)
                    permuted_mi_values.append(permuted_mi)

                mean_permuted_mi = np.mean(permuted_mi_values)
                std_permuted_mi = np.std(permuted_mi_values)
                p_value = np.sum(permuted_mi_values >= mi) / self.stat_sig_perm_num
                estimations[epo_i] = [mi, mean_permuted_mi, std_permuted_mi, p_value]
            else:
                estimations[epo_i, 0] = mi

        return estimations

    def __mi_symb(self, s1: np.ndarray, s2: np.ndarray, l: int = 1, k: int = 3) -> float:
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



        estimations = np.zeros((self._n_epo, 4)) # stores MI result, mean, std, p-value per epoch
        
        for epo_i in range(self._n_epo):

            x, y = (s1[epo_i, :], s2[epo_i, :])
            mi = symb_calc_mi(x, y, l, k)

            if self.calc_sigstats:
                permuted_mi_values = []

                for _ in range(self.stat_sig_perm_num):
                    np.random.shuffle(y)
                    permuted_mi = symb_calc_mi(x, y, l, k)
                    permuted_mi_values.append(permuted_mi)

                mean_permuted_mi = np.mean(permuted_mi_values)
                std_permuted_mi = np.std(permuted_mi_values)
                p_value = np.sum(permuted_mi_values >= mi) / self.stat_sig_perm_num
                estimations[epo_i] = [mi, mean_permuted_mi, std_permuted_mi, p_value]
            else:
                estimations[epo_i, 0] = mi

        return estimations


    def __which_estimator(self, measure: str) -> None:

        self.estimator_name, calculator, properties, initialise_parameter = set_estimator(self.estimator_type, measure, self.params) # from utils.py function

        if calculator:
            self.Calc = calculator()

        if properties:
            for key, value in properties.items():
                self.Calc.setProperty(key, value)

        if initialise_parameter:
            self._initialise_parameter = initialise_parameter


    def __estimate_it(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """ Estimates Mutual Information or Transfer Entropy for a pair of time-series signals using JIDT estimators. """

        estimations = np.zeros((self._n_freq_bands, self._n_epo, 4)) # stores MI/TE result, mean, std, p-value per epoch

        for freq_band in range(self._n_freq_bands):
            for epo_i in range(self._n_epo):
                
                X, Y = (s1[freq_band, epo_i, ...], s2[freq_band, epo_i, ...]) 

                ## GROUPWISE; multivariate time series comparison
                if self._scale_of_organisation > 1:
                    X, Y = X.T, Y.T # transpose to shape (samples, group_channels)

                # Initialise parameter describes the dimensions of the data
                self.Calc.initialise(*self._initialise_parameter) if self._initialise_parameter else self.Calc.initialise()
                self.Calc.setObservations(setup_JArray(X), setup_JArray(Y))
                result = self.Calc.computeAverageLocalOfObservations() * np.log(2)

                if self.calc_sigstats:
                    stat_sig = self.Calc.computeSignificance(self.stat_sig_perm_num)
                    estimations[freq_band, epo_i] = [result, stat_sig.getMeanOfDistribution(), stat_sig.getStdOfDistribution(), stat_sig.pValue]
                else:
                    estimations[freq_band, epo_i, 0] = result
            
        return estimations
    


    def compute_mi(self, estimator_type: str = 'kernel', calc_sigstats: bool = False, vis: bool = False, **kwargs) -> np.ndarray:
        """Function to compute mutual information between data (time-series signals) instantiated in the HyperIT object.

        PARAMETER OPTIONS FOR MUTUAL INFORMATION ESTIMATORS (defaults in parentheses):
        Estimator types:        kwargs:
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
        
        self.measure = 'Mutual Information'
        if estimator_type.lower() == 'ksg':
            estimator_type = 'ksg1'
        self.estimator_type: str = estimator_type.lower()
        self.calc_sigstats: bool = calc_sigstats
        self.vis: bool = vis
        self.params = kwargs

        self.stat_sig_perm_num = self.params.get('stat_sig_perm_num', 100)
        self.p_threshold = self.params.get('p_threshold', 0.05)
        
        # self.__which_mi_estimator()
        self.__which_estimator(measure = 'mi')

        self.mi_matrix = np.zeros((self._n_freq_bands, self._n_epo, self._n_chan, self._n_chan, 4)) if self._scale_of_organisation == 1 else np.zeros((self._n_freq_bands, self._n_epo, self._soi_groups, self._soi_groups, 4))

        loop_range = self._n_chan if self._scale_of_organisation == 1 else self._soi_groups

        for i in tqdm(range(loop_range)):
            for j in range(loop_range):

                if self._inter_brain or i != j:

                    if self._scale_of_organisation == 1:
                        s1, s2 = (self._data1[:, :, i, :], self._data2[:, :, j, :])
                    
                    elif self._scale_of_organisation > 1:
                        s1, s2 = (self._data1[:, :, self._roi[0][i], :], self._data2[:, :, self._roi[1][j], :])

                    if self.estimator_type == 'histogram':
                        self.mi_matrix[:, :, i, j] = self.__mi_hist(s1, s2)
                    elif self.estimator_type == 'symbolic':
                        self.mi_matrix[:, :, i, j] = self.__mi_symb(s1, s2)
                    else:
                        self.mi_matrix[:, :, i, j] = self.__estimate_it(s1, s2)

                    if not self._inter_brain:
                        self.mi_matrix[:, :, j, i] = self.mi_matrix[i, j]


        mi = np.array((self.mi_matrix))

        if self.vis:
            self.__plot_it(mi)

        return mi
    
    def compute_te(self, estimator_type: str = 'kernel', calc_sigstats: bool = False, vis: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Function to compute transfer entropy between data (time-series signals) instantiated in the HyperIT object. 
            data1 is taken to be the source and data2 the target (X -> Y). This function automatically computes the opposite matrix, too (Y -> X).

        PARAMETER OPTIONS FOR TRANSFER ENTROPY ESTIMATORS (defaults in parentheses):
        Estimator types:        kwargs:
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
        
        self.measure = 'Transfer Entropy'
        self.estimator_type: str = estimator_type.lower()
        self.calc_sigstats: bool = calc_sigstats
        self.vis: bool = vis
        self.params = kwargs
        
        self.stat_sig_perm_num = self.params.get('stat_sig_perm_num', 100)
        self.p_threshold = self.params.get('p_threshold', 0.05)

        # self.estimator = self.__which_te_estimator()
        self.__which_estimator(measure = 'te')

        self.te_matrix_xy, self.te_matrix_yx = (np.zeros((self._n_freq_bands, self._n_epo, self._n_chan, self._n_chan, 4)), 
                                                np.zeros((self._n_freq_bands, self._n_epo, self._n_chan, self._n_chan, 4))) if self._scale_of_organisation == 1 else (np.zeros((self._n_freq_bands, self._n_epo, self._soi_groups, self._soi_groups, 4)), 
                                                                                                                                                                  np.zeros((self._n_freq_bands, self._n_epo, self._soi_groups, self._soi_groups, 4)))

        loop_range = self._n_chan if self._scale_of_organisation == 1 else self._soi_groups

        for i in tqdm(range(loop_range)):
            for j in range(loop_range):
                
                if self._inter_brain or i != j: # avoid self-channel calculations for intra_brain condition
                    
                    if self._scale_of_organisation == 1:
                        s1, s2 = (self._data1[:, :, i, :], self._data2[:, :, j, :]) 
                    
                    elif self._scale_of_organisation > 1:
                        s1, s2 = (self._data1[:, :, self._roi[0][i], :], self._data2[:, :, self._roi[1][j], :]) 

                    self.te_matrix_xy[:, :, i, j] = self.__estimate_it(s1, s2)
                    
                    if self._inter_brain: # don't need to compute opposite matrix for intra-brain as we already loop through each channel combination including symmetric
            
                        self.te_matrix_yx[:, :, i, j] = self.__estimate_it(s2, s1)
                    
        te_xy = np.array((self.te_matrix_xy))
        te_yx = np.array((self.te_matrix_yx))
        
        if self.vis:
            print("Plotting Transfer Entropy for X -> Y...")
            self.__plot_it(te_xy)
            if self._inter_brain:
                print("Plotting Transfer Entropy for Y -> X...")
                self.__plot_it(te_yx)
                
        return te_xy, te_yx

    def compute_atoms(self, tau: int = 1, redundancy: str = 'MMI', vis: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Function to compute Integrated Information Decomposition (ΦID) between data (time-series signals) instantiated in the HyperIT object.
            Option to visualise the lattice values for a specific channel pair (be sure to specify via plot_channels kwarg).

        Args:
            tau             (int, optional): Time-lag parameter. Defaults to 1.
            kind            (str, optional): Estimator type. Defaults to "gaussian".
            redundancy      (str, optional): Redundancy function to use. Defaults to 'MMI' (Minimum Mutual Information).
            vis            (bool, optional): Whether to visualise (via __plot_atoms()). Defaults to False.

        Returns:
              Tuple(np.ndarray, np.ndarray): Two matrices of Integrated Information Decomposition dictionaries (representing all atoms, both X->Y and Y->X), each with shape (n_chan, n_chan),
        """
        
        self.measure = 'Integrated Information Decomposition'

        loop_range = self._n_chan if self._scale_of_organisation == 1 else self._soi_groups

        phi_dict_xy = [[[{} for _ in range(loop_range)] for _ in range(loop_range)] for _ in range(self._n_freq_bands)]
        phi_dict_yx = [[[{} for _ in range(loop_range)] for _ in range(loop_range)] for _ in range(self._n_freq_bands)]


        for freq_band in range(self._n_freq_bands):
            for i in tqdm(range(loop_range)):
                for j in range(loop_range):
                    
                    if self._inter_brain or i != j:

                        if self._scale_of_organisation == 1:
                            s1, s2 = (self._data1[freq_band, :, i, :], self._data2[freq_band, :, j, :]) 
                            if self._n_epo > 1:
                                s1, s2 = s1.reshape(-1), s2.reshape(-1)

                                ## If you want to pass (samples, epochs) as atomic calculations, delete line above and uncomment line below
                                # s1, s2 = s1.T, s2.T
                        
                        elif self._scale_of_organisation > 1:
                            
                            temp_s1, temp_s2 = self._data1[freq_band, :, self._roi[0][i], :], self._data2[freq_band, :, self._roi[1][j], :]
                            # Flatten epochs and transpose to shape (samples, channels) [necessary configuration for phyid]
                            s1, s2 = temp_s1.transpose(1,0,2).reshape(-1, temp_s1.shape[1]), temp_s2.transpose(1,0,2).reshape(-1, temp_s2.shape[1])
                                

                        atoms_results, _ = calc_PhiID(s1, s2, tau=tau, kind='gaussian', redundancy=redundancy)
                        calc_atoms = np.mean(np.array([atoms_results[_] for _ in PhiID_atoms_abbr]), axis=1)
                        phi_dict_xy[freq_band][i][j] = {key: value for key, value in zip(atoms_results.keys(), calc_atoms)}

                        if self._inter_brain:
                            atoms_results, _ = calc_PhiID(s2, s1, tau=tau, kind='gaussian', redundancy=redundancy)
                            calc_atoms = np.mean(np.array([atoms_results[_] for _ in PhiID_atoms_abbr]), axis=1)
                            phi_dict_yx[freq_band][i][j] = {key: value for key, value in zip(atoms_results.keys(), calc_atoms)}   

        if vis:
            self.__plot_atoms(phi_dict_xy)
            self.__plot_atoms(phi_dict_yx) 

        return phi_dict_xy, phi_dict_yx
    



    def __plot_it(self, it_matrix: np.ndarray) -> None:
        """Plots the Mutual Information or Transfer Entropy matrix for visualisation. 
        Axes labelled with source and target channel names. 
        Choice to plot for all epochs, specific epoch(s), or average across epochs.

        Args:
            it_matrix (np.ndarray): The Mutual Information or Transfer Entropy matrix to be plotted with shape (n_chan, n_chan, n_epo, 4), 
            where the last dimension represents the statistical signficance testing results: (local result, distribution mean, distribution standard deviation, p-value).
        """

        title = f'{self.measure} | {self.estimator_name} \n {"Inter-Brain" if self._inter_brain else "Intra-Brain"}'
        epochs = [0] # default to un-epoched or epoch-average case
        choice = None
        
        if self._scale_of_organisation > 1:
            source_channel_names = convert_indices_to_names(self._channel_names, self._channel_indices1, 0) if self._scale_of_organisation == 1 else convert_indices_to_names(self._channel_names, self._roi[0], 0)
            target_channel_names = convert_indices_to_names(self._channel_names, self._channel_indices2, 1) if self._scale_of_organisation == 1 else convert_indices_to_names(self._channel_names, self._roi[1], 1)

            print("Plotting for grouped channels.")

            print("Source Groups:")
            for i in range(self._soi_groups):
                print(f"{i+1}: {source_channel_names[i]}")

            print("\n\nTarget Groups:")
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

    def __plot_atoms(self, phi_dict: dict):
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





if __name__ == '__main__':
    # Example usage
    data1 = np.random.randn(31, 1000)
    data2 = np.random.randn(31, 1000)
    channel_names = [['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'Fz', 'FT9', 'FT10', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'], 
                     ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'Fz', 'FT9', 'FT10', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']]

    import os
    hyperit = HyperIT(data1, data2, channel_names, standardise_data=True, verbose=True)

    mi = hyperit.compute_mi(estimator_type='ksg', calc_sigstats=False, vis=True)

    hyperit.roi = [[['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'Fz', 'FT9', 'FT10', 'FC5', 'FC1', 'FC2', 'FC6'],
                    ['T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3']], 
                    [['TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'],
                    ['T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3']]]
    

    mi = hyperit.compute_te(estimator_type='ksg', calc_sigstats=False, vis=True, k=5, k_tau=4, l=2, delay=10, normalise=False)


    # mi = hyperit.compute_mi(estimator_type='kernel', calc_sigstats=True, vis=True)
    # te_xy, te_yx = hyperit.compute_te(estimator_type='kernel', calc_sigstats=True, vis=True)
    # phi_xy, phi_yx = hyperit.compute_atoms(tau=1, redundancy='mmi', vis=True, plot_channels=[0, 0])