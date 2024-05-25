import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from typing import Tuple, List, Union, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pkg_resources import resource_filename

from jpype import isJVMStarted, startJVM, getDefaultJVMPath
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

from .utils import (
    setup_JArray, 
    ensure_three_dims,
    bandpass_filter_data, 
    convert_names_to_indices, 
    convert_indices_to_names, 
    set_estimator
)

from enum import Enum

class MeasureType(Enum):
    MI = 'mi' # Mutual Information
    TE = 'te' # Transfer Entropy
    PhyID = 'phyid' # Integrated Information Decomposition

    def __str__(self):
        return self.value  


class HyperIT:
    """
    HyperIT: Hyperscanning Analyses using Information Theoretic Measures.

    HyperIT is equipped to compute pairwise and multivariate Mutual Information (MI), Transfer Entropy (TE),
    and Integrated Information Decomposition (ΦID) for continuous time-series data. Compatible for both
    intra-brain and inter-brain analyses and for both epoched and unepoched data. Analyses can be conducted
    at specified frequency bands (via bandpass filtering) and option to standardise data before computing measures.
    
    Multiple estimator choices and parameter customisations (via JIDT) are available, including KSG, Kernel, Gaussian,
    Symbolic, and Histogram/Binning. 
    
    Integrated statistical significance testing using permutation/boostrapping approach.
    
    Visualisations of MI/TE matrices also provided.

    Args:
        data1                       (np.ndarray): Time-series data for participant 1. Can take shape (n_epo, n_chan, n_samples) or (n_chan, n_samples) for epoched and unepoched data, respectively.
        data2                       (np.ndarray): Time-series data for participant 2. Must have the same shape as data1. 
        channel_names      (List[str], optional): A list of strings representing the channel names for each participant. [[channel_names_p1], [channel_names_p2]] or [[channel_names_p1]].
        sfreq                  (float, optional): Sampling frequency of the data.
        freq_bands              (dict, optional): Dictionary of frequency bands for bandpass filtering. {band_name: (low_freq, high_freq)}.
        standardise_data        (bool, optional): Whether to standardise the data before analysis. Defaults to True.
        verbose                 (bool, optional): Whether constructor and analyses should output details and progress. Defaults to False.
        show_tqdm               (bool, optional): Whether to output the tqdm progress bar in the console output. Defaults to True (recommended).
        **filter_options        (dict, optional): Additional keyword arguments for bandpass filtering.

    Note:
        This class requires numpy, mne, matplotlib, jpype (with the local infodynamics.jar file), and phyid as dependencies.
        
        Before a HyperIT can be created, users must first call HyperIT.setup_JVM() to initialise the Java Virtual
        Machine (JVM) with the local directory location of the infodynamics.jar file. Users can then create multiple HyperIT
        objects containing time-series data, later calling various functions for analysis. 
        
        Automatic data checks for consistency and dimensionality, identifying whether analysis is to be intra- or inter-brain 
        and epochality of data.
            - If data is 3 dimensional, data is assumed to be epoched with shape (epochs, channels, time_points).
            - If data is 2 dimensional, data is assumed to be unepoched with shape (channels, time_points).
            - If data is 1 dimensional, data is assumed to be single channel time series with shape (time_points).
    """


    ## SETTING UP JVM

    _jvm_initialised = False

    @classmethod
    def setup_JVM(cls, verbose: bool = False) -> None:
        """Setup JVM if not already started. To be called once before creating any instances."""
        if not cls._jvm_initialised:
            if not isJVMStarted():
                jarLocation = resource_filename(__name__, 'infodynamics.jar')
                startJVM(getDefaultJVMPath(), "-ea", ('-Djava.class.path=' + jarLocation))
                cls._jvm_initialised = True
                if verbose:
                    print("JVM started successfully.")
            else:
                if verbose:
                    print("JVM already started.")
        else:
            if verbose:
                print("JVM setup already completed.")



    def __init__(self, data1: np.ndarray, data2: np.ndarray, channel_names: List[str] = None, sfreq: float = None, freq_bands: dict = None, standardise_data: bool = False, verbose: bool = False, show_tqdm = True, **filter_options) -> None:

        if not self.__class__._jvm_initialised:
            raise RuntimeError("JVM has not been started. Call setup_JVM() before creating any instances of HyperIT.")

        self._verbose: bool = verbose
        self._channel_names = channel_names or None   
            
        self._channel_indices1 = []
        self._channel_indices2 = []
        self._sfreq = sfreq or None
        self._freq_bands = freq_bands or None
        self._filter_options = filter_options
        self._standardise_data = standardise_data
        self._show_tqdm = show_tqdm

        self._data1 = data1
        self._data2 = data2

        self.__setup()

        # Store original data (that has been checked and manipulated) for resetting
        self._orig_channel_indices_1 = self._channel_indices1
        self._orig_channel_indices_2 = self._channel_indices2

        if self._verbose:
            print("HyperIT object created successfully.")

            if self._n_epo > 1:
                print(f"{'Inter-Brain' if self._hyper else 'Intra-Brain'} analysis and epoched data detected. \nAssuming data passed have shape ({self._n_epo} epochs, {self._n_chan} channels, {self._n_samples} time points).")
            else:
                print(f"{'Inter-Brain' if self._hyper else 'Intra-Brain'} analysis and unepoched data detected. \nAssuming data passed have shape ({self._n_chan} channels, {self._n_samples} time points).")

            if self._freq_bands:
                print(f"Data has been bandpass filtered: {self._freq_bands}.")


    def __setup(self):
        self.__check_data()
        self.__check_channels()
        self.__configure_data()

        _, self._n_epo, self._n_freq_bands, self._n_chan, self._n_samples = self._all_data.shape

        self._it_matrix = None
        self._initialise_parameter = None

        # These are default when HyperIT object is instantiated. 
        self._roi = []
        self._scale_of_organisation = 1 # 1 = micro organisation (single channel pairwise), n = meso- or n-scale organisation (n-sized groups of channels)
        

        

    ## DUNDER METHODS

    def __repr__(self) -> str:
        """ String representation of HyperIT object. """
        analysis_type = 'Hyperscanning' if self._hyper else 'Intra-Brain'
        channel_info = f"{self._channel_names[0]}"  # Assuming self._channel_names[0] is a list of channel names for the first data set
        
        # Adding second channel name if interbrain analysis is being conducted
        if self._hyper:
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

        if self._data2 is None or np.array_equal(self._data1, self._data2) or self._data2.shape == (0,):
            self._data2 = self._data1.copy()
            self._hyper = False
            self._include_intra = False
        else:
            self._hyper = True
            
        if not all(isinstance(data, np.ndarray) for data in [self._data1, self._data2]):
            raise ValueError("Time-series data must be numpy arrays.")
        
        if self._data1.shape != self._data2.shape:
            raise ValueError("Time-series data must have the same shape for both participants.")
    
        if self._data1.ndim not in [1,2,3]:
            raise ValueError(f"Unexpected number of dimensions in time-series data: {self._data1.ndim}. Expected 3 dimensions (epochs, channels, time_points) or 2 dimensions (channels, time_points) or 1 dimension (time_points).")

        # Ensure data is 3 dimensional and has shape (n_epochs, n_channels, n_samples). 
        # If freq_bands have been specified, this will become 4 dimensional: (n_epochs, n_freq_bands, n_channels, n_samples)
        self._data1, self._data2 = map(ensure_three_dims, (self._data1, self._data2))

    def __check_channel_names(self) -> None:
        """ Checks the consistency of the channel names provided. """

        if not isinstance(self._channel_names, list):
            raise ValueError("Channel names must be a list of strings or a list of lists of strings.")

        elif isinstance(self._channel_names[0], str):
            if not all(isinstance(name, str) for name in self._channel_names):
                raise ValueError("All elements must be strings if the first element is a string.")
            self._channel_names = [self._channel_names, self._channel_names.copy()]

        elif isinstance(self._channel_names[0], list):
            if not all(isinstance(sublist, list) for sublist in self._channel_names):
                raise ValueError("All elements must be lists of strings if the first element is a list.")
            for sublist in self._channel_names:
                if not all(isinstance(name, str) for name in sublist):
                    raise ValueError("All sublists must be lists of strings.")
            if len(self._channel_names) == 1:
                self._channel_names = self._channel_names * 2
        else:
            raise ValueError("Channel names must be either a list of strings or a list of lists of strings.")

        if any(len(names) != self._n_chan for names in self._channel_names):
            raise ValueError("The number of channels in time-series data does not match the length of channel_names.")

    def __check_channels(self) -> None:
        """ Checks the consistency of the channel names provided and sets the number of channels as an object variable. """

        self._n_chan = self._data1.shape[1] 
        self._channel_indices1, self._channel_indices2 = np.arange(self._n_chan), np.arange(self._n_chan)
        
        if self._channel_names:
            self.__check_channel_names()
        else:
            self._channel_names = [np.arange(self._n_chan), np.arange(self._n_chan)]
            
    def __configure_data(self) -> None:
        """ Configures the data for analysis by bandpass filtering and standardising. """

        if self._freq_bands:
            self._data1, self._data2 = bandpass_filter_data(self._data1, self._sfreq, self._freq_bands, **self._filter_options), bandpass_filter_data(self._data2, self._sfreq, self._freq_bands, **self._filter_options)
        else:
            self._data1, self._data2 = np.expand_dims(self._data1, axis=1), np.expand_dims(self._data2, axis=1)

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

        ## DETERMINE SCALE OF ORGANISATION
        # 1: Micro organisation (specified channels, pairwise comparison)           e.g., roi_list = [['Fp1', 'Fp2'], ['F3', 'F4']]
        # n: Meso- or n-scale organisation (n specified channels per ROI group)     e.g., roi_list = [[  ['Fp1', 'Fp2'], ['CP1', 'CP2']   ],   n CHANNELS IN EACH GROUP FOR PARTICIPANT 1
        #                                                                                             [  ['Fp1', 'Fp2'], ['F3', 'F4']     ]].

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
        
        if self._verbose:
            print(f"Scale of organisation: {self._scale_of_organisation} channels.")
            print(f"Groups of channels: {self._soi_groups}")

        self._channel_indices1, self._channel_indices2 = [convert_names_to_indices(self._channel_names, part, idx) for idx, part in enumerate(roi_list)] 

        self._roi = [self._channel_indices1, self._channel_indices2]
        
    def reset_roi(self) -> None:
        """Resets the region of interest for both data of the HyperIT object to all channels."""
        self._scale_of_organisation = 1
        self._initialise_parameter = None
        self._channel_indices1, self._channel_indices2 = self._orig_channel_indices_1, self._orig_channel_indices_2
        self._roi = []


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

        if self._calc_statsig:
            permuted_mi_values = []

            for _ in range(self._stat_sig_perm_num):
                np.random.shuffle(s2)
                permuted_mi = hist_calc_mi(s1, s2)
                permuted_mi_values.append(permuted_mi)

            mean_permuted_mi = np.mean(permuted_mi_values)
            std_permuted_mi = np.std(permuted_mi_values)
            p_value = np.sum(permuted_mi_values >= result) / self._stat_sig_perm_num
            return np.array([result, mean_permuted_mi, std_permuted_mi, p_value])

        return result

    def __estimate_mi_symb(self, s1: np.ndarray, s2: np.ndarray, k: int = 3, delay: int = 1) -> float:
        """Calculates Mutual Information using Symbolic Estimator for time-series signals."""

        symbol_weights = np.power(k, np.arange(k))

        def symb_symbolise(X: np.ndarray, k: int, delay: int) -> np.ndarray:
            Y = np.empty((k, len(X) - (k - 1) * delay))
            for i in range(k):
                Y[i] = X[i * delay:i * delay + Y.shape[1]]
            return Y.T

        def symb_normalise_counts(d) -> None:
            total = sum(d.values())        
            return {key: value / total for key, value in d.items()}
        
        def symb_calc_mi(X: np.ndarray, Y: np.ndarray, k: int, delay: int) -> float:
            X = symb_symbolise(X, delay, k).argsort(kind='quicksort')
            Y = symb_symbolise(Y, delay, k).argsort(kind='quicksort')

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


        result = symb_calc_mi(s1, s2, k, delay)

        if self._calc_statsig:
            permuted_mi_values = []

            for _ in range(self._stat_sig_perm_num):
                np.random.shuffle(s2)
                permuted_mi = symb_calc_mi(s1, s2, k, delay)
                permuted_mi_values.append(permuted_mi)

            mean_permuted_mi = np.mean(permuted_mi_values)
            std_permuted_mi = np.std(permuted_mi_values)
            p_value = np.sum(permuted_mi_values >= result) / self._stat_sig_perm_num
            return np.array([result, mean_permuted_mi, std_permuted_mi, p_value])

        return result



    ## ESTIMATION AND HELPER FUNCTIONS

    def __which_estimator(self, measure: str) -> None:
        """Determines the estimator to use based on the measure type and sets the estimator, properties, and initialisation parameters."""

        self._estimator_name, calculator, properties, initialise_parameter = set_estimator(self._estimator, measure, self._params) # from utils.py function

        if calculator:
            self._Calc = calculator()

        if properties:
            for key, value in properties.items():
                self._Calc.setProperty(key, value)

        if initialise_parameter:
            self._initialise_parameter = initialise_parameter

    def __setup_matrix(self) -> None:
        """ Sets up the matrices for Mutual Information, Transfer Entropy, or Integrated Information Decomposition. """

        # POINTWISE CHANNEL COMPARISON (If ROI is selected, this will pick specific channels, otherwise data stays the same)
        if self._scale_of_organisation == 1:
            self._it_data1 = self._data1[:, :, self._channel_indices1, :]
            self._it_data2 = self._data2[:, :, self._channel_indices2, :]
            self._n_chan = len(self._channel_indices1)
        else:
            self._it_data1 = self._data1.copy()
            self._it_data1 = self._data2.copy()

        self._loop_range = self._n_chan if self._scale_of_organisation == 1 else self._soi_groups
        

        if self._hyper and self._include_intra:
            self._loop_range *= 2 
            # Will give shape (n_epo, n_freq_bands, n_chan_or_group*2, n_samples)
            # Note that data1 and data2 will be identical
            temp1 = self._it_data1.copy()
            temp2 = self._it_data2.copy()
            self._it_data1 = np.concatenate((temp1, temp2), axis=2)
            self._it_data2 = np.concatenate((temp1, temp2), axis=2)

            if self._scale_of_organisation != 1:
                temp_list = [self._roi[0],[[item + self._n_chan for item in sublist] for sublist in self._roi[1]]]
                self._roi = [sublist for outer_list in temp_list for sublist in outer_list]

        if self._measure == MeasureType.MI or self._measure == MeasureType.TE:

            if self._calc_statsig:
                self._it_matrix = np.zeros((self._n_epo, self._n_freq_bands, self._loop_range, self._loop_range, 4))
                return
            
            self._it_matrix = np.zeros((self._n_epo, self._n_freq_bands, self._loop_range, self._loop_range))
            return
        
        self._it_matrix = np.zeros((self._n_epo, self._n_freq_bands, self._loop_range, self._loop_range, 16))
        

    def __estimate_it(self, s1: np.ndarray, s2: np.ndarray) -> float | np.ndarray:
        """ Estimates Mutual Information or Transfer Entropy for a pair of time-series signals using JIDT estimators. """

        # Initialise parameter describes the dimensions of the data
        self._Calc.initialise(*self._initialise_parameter) if self._initialise_parameter else self._Calc.initialise()
        self._Calc.setObservations(setup_JArray(s1), setup_JArray(s2))
        result = self._Calc.computeAverageLocalOfObservations() * np.log(2)

        # Conduct significance testing
        if self._calc_statsig:
            stat_sig = self._Calc.computeSignificance(self._stat_sig_perm_num)
            return np.array([result, stat_sig.getMeanOfDistribution(), stat_sig.getStdOfDistribution(), stat_sig.pValue])
            
        return float(result)
    
    def __estimate_atoms(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """ Estimates Integrated Information Decomposition for a pair of time-series signals using phyid package. """
        
        try:
            atoms_results_xy, _ = calc_PhiID(s1, s2, tau=self._tau, kind='gaussian', redundancy=self._redundancy)
        except Exception as e:
            print(f'Warning: error handling timeseries. They are likely identical or similar timeseries. Setting results to 0. Error: {e}', flush=True)
            return np.zeros(16)
            
        calc_atoms_xy = np.mean(np.array([atoms_results_xy[_] for _ in PhiID_atoms_abbr]), axis=1)
        return calc_atoms_xy
        

    def __filter_estimation(self, s1: np.ndarray, s2: np.ndarray) -> float | np.ndarray:
        """ Filters the estimation in case incompatible with JIDT. """

        if self._measure == MeasureType.MI:
            match self._estimator:
                case 'histogram':
                    return self.__estimate_mi_hist(s1, s2)
                case 'symbolic':
                    return self.__estimate_mi_symb(s1, s2, self._params.get('k', 3), self._params.get('delay', 1))

        elif self._measure == MeasureType.PhyID:
            return self.__estimate_atoms(s1, s2)

        return self.__estimate_it(s1, s2)

    def __compute_pair_or_group(self, epoch: int, freq_band: int, i: int, j: int) -> None:
        """ Computes the Mutual Information or Transfer Entropy for a pair of channels or groups of channels. """

        channel_or_group_i = i if self._scale_of_organisation == 1 else self._roi[i]
        channel_or_group_j = j if self._scale_of_organisation == 1 else self._roi[j]    
        
        # Data needs to have shape (samples, channels/groups), if not pointwise comparison 
        # (this is how both JIDT and phyid handle and expect incoming multivariate data)
        # (Note that .T does not affect pointwise comparison as it is already in the correct shape)
        s1, s2 = self._it_data1[epoch, freq_band, channel_or_group_i, :].T, self._it_data2[epoch, freq_band, channel_or_group_j, :].T

        if ((not self._hyper) or self._include_intra) and not i==j:
            
            if self._measure == MeasureType.MI and j < i:
                result = self.__filter_estimation(s1, s2)
                self._it_matrix[epoch, freq_band, i, j] = result
                self._it_matrix[epoch, freq_band, j, i] = result
                return
                
            self._it_matrix[epoch, freq_band, i, j] = self.__filter_estimation(s1, s2)
            return

        if self._measure == MeasureType.MI and j <= i:
            result = self.__filter_estimation(s1, s2)
            self._it_matrix[epoch, freq_band, i, j] = result
            self._it_matrix[epoch, freq_band, j, i] = result
            return
            
        self._it_matrix[epoch, freq_band, i, j] = self.__filter_estimation(s1, s2)
        return


    ## MAIN CALCULATION FUNCTIONS         

    def __main_calc(self) -> np.ndarray:

        if self._include_intra and not self._hyper:
            raise ValueError("Intra-brain analyses are only available for hyperscanning set-ups. Identical data or only one set of data was detected.")

        self.__setup_matrix()

        with tqdm(total=self._n_epo * self._n_freq_bands * self._loop_range, disable=not self._show_tqdm) as tqdm_bar:
            for epoch in range(self._n_epo):
                for freq_band in range(self._n_freq_bands):
                    tqdm_bar.set_description(f"Computing Epoch {epoch+1}/{self._n_epo} | Frequency Band {list(self._freq_bands.keys())[freq_band]}")
                    for i in range(self._loop_range):
                        for j in range(self._loop_range):
                            self.__compute_pair_or_group(epoch, freq_band, i, j)
                        tqdm_bar.update(1)
                
        if self._vis:
            self.__prepare_vis()

        return self._it_matrix

    def __setup_mite_calc(self, estimator: str, include_intra: bool, calc_statsig: bool, stat_sig_perm_num: int, p_threshold: float, vis: bool, plot_epochs: List, **kwargs) -> np.ndarray:
        """ General function for computing Mutual Information or Transfer Entropy. """

        self._estimator = estimator.lower() if not (self._measure == MeasureType.MI and estimator.lower() == 'ksg') else 'ksg1'
        self._calc_statsig = calc_statsig
        self._include_intra = include_intra
        self._vis = vis
        self._plot_epochs = (plot_epochs or None) if self._vis else None
        self._params = kwargs
        self._stat_sig_perm_num = stat_sig_perm_num
        self._p_threshold = p_threshold
        
        # Set up the estimator and properties
        self.__which_estimator(str(self._measure))
        
        return self.__main_calc()

    def __setup_atom_calc(self, tau: int, redundancy: str, include_intra: bool) -> np.ndarray:
        """ General function for computing Integrated Information Decomposition. """

        self._tau = tau
        self._redundancy = redundancy
        self._include_intra = include_intra
        self._vis = False

        return self.__main_calc()



    ## VISUALISATION FUNCTIONS

    def __prepare_vis(self) -> None:
        """ Prepares the visualisation of Mutual Information, Transfer Entropy, or Integrated Information Decomposition matrix/matrices. """

        if self._plot_epochs is None or self._plot_epochs == [-1]:
            self._plot_epochs = range(self._n_epo)
        else:
            self._plot_epochs = [ep - 1 for ep in self._plot_epochs if ep < self._n_epo]
            if not self._plot_epochs:
                raise ValueError("No valid epochs found in the input list.")
            
        if self._n_chan == 1:
            print("Single channel detected. No visualisation possible.")
            return

        if self._verbose:
            print(f"Plotting {self._measure_title}...")
    
        self.__plot_it()

    def __plot_it(self) -> None:
        """Plots the Mutual Information or Transfer Entropy matrix for visualisation. 
        Axes labelled with source and target channel names. 
        """

        title = f'{self._measure_title} | {self._estimator_name} \n'
        source_channel_names = convert_indices_to_names(self._channel_names, self._channel_indices1, 0) 
        target_channel_names = convert_indices_to_names(self._channel_names, self._channel_indices2, 1)

        global_max = np.max(self._it_matrix[..., 0]) if self._calc_statsig else np.max(self._it_matrix)

        if self._scale_of_organisation > 1:
            
            print("Plotting for grouped channels.")
            print("Source Groups:")
            source_str = []
            for i in range(self._soi_groups):
                source_str.append(f'X{i+1}_{source_channel_names[i]}')
                print(f"{i+1}: {source_channel_names[i]}")

            print("\nTarget Groups:")
            target_str = []
            for i in range(self._soi_groups):
                target_str.append(f'Y{i+1}_{target_channel_names[i]}')
                print(f"{i+1}: {target_channel_names[i]}")

        for epoch in self._plot_epochs:
            for freq_band in range(self._n_freq_bands):
                results = self._it_matrix[epoch, freq_band, :, :, 0] if self._calc_statsig else self._it_matrix[epoch, freq_band, :, :]
                
                plt.figure(figsize=(12, 10))
                img = plt.imshow(results, cmap='BuPu', vmin=0, vmax=global_max, aspect='auto')

                band_description = ""
                if self._freq_bands:
                    band_name, band_range = list(self._freq_bands.items())[freq_band]
                    band_description = f"{band_name}: {band_range}"

                if self._n_epo > 1:
                    if band_description:
                        plt.title(f'{title} Epoch {epoch+1}, Frequency Band {band_description}', pad=20)
                    else:
                        plt.title(f'{title} Epoch {epoch+1}', pad=20)
                else:
                    if band_description:
                        plt.title(f'{title} Frequency Band {band_description}', pad=20)
                    else:
                        plt.title(title, pad=20)
                    
                if self._calc_statsig:
                    for i in range(self._loop_range):
                        for j in range(self._loop_range):
                            if i == j:
                                continue
                            p_val = float(self._it_matrix[epoch, freq_band, i, j, 3])
                            if p_val < self._p_threshold:
                                normalised_value = (self._it_matrix[epoch, freq_band, i, j, 0] - np.min(results)) / (np.max(results) - np.min(results))
                                text_colour = 'white' if normalised_value > 0.5 else 'black'
                                plt.text(j, i, f'p={p_val:.2f}', ha='center', va='center', color=text_colour, fontsize=8, fontweight='bold')


                cbar = plt.colorbar(img)
                cbar.set_label(self._measure_title, rotation=270, labelpad=20)
                n_ticks = 8
                ticks = np.linspace(0, global_max, n_ticks)
                # ticks = list(cbar.get_ticks())
                # if global_max not in ticks:
                #     ticks.append(global_max)
                # ticks = sorted(set(ticks))
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{tick:.2f}" if tick != global_max else f"{tick:.2f} (max)" for tick in ticks])

                x_tick_label = target_channel_names.copy()
                y_tick_label = source_channel_names.copy()
                ticks = range(self._loop_range)
                rotate_x = 90 
                rotate_y = 0 

                if self._hyper and self._include_intra:
                    if self._scale_of_organisation != 1:
                        x_tick_label = str(source_str) + str(target_str)
                    else:
                        x_tick_label = ['X_' + str(s) for s in y_tick_label] + ['Y_' + str(s) for s in x_tick_label]
                    
                    y_tick_label = x_tick_label
                else:
                    plt.xlabel('Target')
                    plt.ylabel('Source')

                plt.xticks(ticks, x_tick_label, rotation=rotate_x) 
                plt.yticks(ticks, y_tick_label, rotation=rotate_y)

                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=True)
                plt.tick_params(axis='y', which='both', right=False, left=True, labelleft=True)
                plt.show()


    ## HIGH-LEVEL INTERFACE FUNCTIONS

    def compute_mi(self, estimator: str = 'kernel', include_intra: bool = False, calc_statsig: bool = False, stat_sig_perm_num: int = 100, p_threshold: float = 0.05, vis: bool = False, plot_epochs: List[int] = None, **kwargs) -> np.ndarray:
        """
        Computes Mutual Information (MI) between data (time-series signals) using specified estimator.

        Args:
            estimator               (str, optional): Specifies the MI estimator to use. Defaults to 'kernel'.
            include_intra          (bool, optional): If True, includes intra-brain analyses. Defaults to False.
            calc_statsig          (bool, optional): If True, performs statistical significance testing. Defaults to False.
            vis                    (bool, optional): If True, results will be visualised. Defaults to False.
            plot_epochs       (List[int], optional): Specifies which epochs to plot. None plots all. Defaults to None.
            **kwargs                               : Additional keyword arguments for the MI estimator.

        Returns:
            np.ndarray: A symmetric mutual information matrix. The shape of the matrix is determined by
            the `include_intra` and `calc_statsig` flags:
                - If `include_intra` is False, shape is (n_epo, n_freq_bands, n_chan, n_chan).
                - If `include_intra` is True, shape is (n_epo, n_freq_bands, 2*n_chan, 2*n_chan).
                - If `calc_statsig` is True, an additional last dimension (size 4) contains statistical
                significance results: [MI value, mean, standard deviation, p-value].

        Note:
            When `include_intra` is True, the matrix can be segmented accordingly:
                - `intra1`: matrix[:, :, :n_chan, :n_chan]
                - `intra2`: matrix[:, :, n_chan:, n_chan:]
                - `inter12`: matrix[:, :, :n_chan, n_chan:]
                - `inter21`: matrix[:, :, n_chan:, :n_chan]

        Available Estimators and Their Parameters:
            - histogram: 
                - None.
            - ksg1:
                - kraskov_param (int, default=4).
                - normalise (bool, default=True).
            - ksg2:
                - kraskov_param (int, default=4).
                - normalise (bool, default=True).
            - kernel:
                - kernel_width (float, default=0.25).
                - normalise (bool, default=True).
            - gaussian:
                - normalise (bool, default=True).
            - symbolic:
                - k (int, default=3): Embedding history or symbol length.
                - delay (int, default=1).
        """
        self._measure = MeasureType.MI
        self._measure_title = 'Mutual Information'
        return self.__setup_mite_calc(estimator, include_intra, calc_statsig, stat_sig_perm_num, p_threshold, vis, plot_epochs, **kwargs)

    def compute_te(self, estimator: str = 'kernel', include_intra: bool = False, calc_statsig: bool = False, stat_sig_perm_num: int = 100, p_threshold: float = 0.05, vis: bool = False, plot_epochs: List[int] = None, **kwargs) -> np.ndarray:
        """
        Computes Transfer Entropy (TE) between time-series data using a specified estimator. This function allows for intra-brain and inter-brain analyses and includes optional statistical significance testing. Data1 is considered the source and Data2 the target.

        Args:
            estimator               (str, optional): Specifies the TE estimator to use. Defaults to 'kernel'.
            include_intra          (bool, optional): Whether to include intra-brain comparisons in the output matrix. Defaults to False.
            calc_statsig          (bool, optional): Whether to calculate statistical significance of TE values. Defaults to False.
            vis                    (bool, optional): Enables visualisation of the TE matrix if set to True. Defaults to False.
            plot_epochs       (List[int], optional): Specifies which epochs to plot. If None, plots all epochs. Defaults to None.
            **kwargs                               : Additional parameters for the TE estimator.

        Returns:
            np.ndarray: A transfer entropy matrix. The shape of the matrix is determined by
            the `include_intra` and `calc_statsig` flags:
                - If `include_intra` is False, shape is (n_epo, n_freq_bands, n_chan, n_chan).
                - If `include_intra` is True, shape is (n_epo, n_freq_bands, 2*n_chan, 2*n_chan).
                - If `calc_statsig` is True, an additional last dimension (size 4) contains statistical
                significance results: [MI value, mean, standard deviation, p-value].
        Note:
            When `include_intra` is True, the matrix can be segmented accordingly:
                - `intra1`: matrix[:, :, :n_chan, :n_chan]
                - `intra2`: matrix[:, :, n_chan:, n_chan:]
                - `inter12`: matrix[:, :, :n_chan, n_chan:]
                - `inter21`: matrix[:, :, n_chan:, :n_chan]

        Available Estimators and Their Parameters:
            - `ksg`:
                - k, k_tau (int, default=1): Target and source embedding history length.
                - l, l_tau (int, default=1): Target and source embedding delay.
                - delay (int, default=1): Delay parameter for temporal dependency.
                - kraskov_param (int, default=1).
                - normalise (bool, default=True).
            - `kernel`:
                - kernel_width (float, default=0.5).
                - normalise (bool, default=True).
            - `gaussian`:
                - k, k_tau (int, default=1): Target and source embedding history length.
                - l, l_tau (int, default=1): Target and source embedding delay.
                - delay (int, default=1): Delay parameter for temporal dependency.
                - bias_correction (bool, default=False).
                - normalise (bool, default=True).
            - `symbolic`:
                - k (int, default=1): Embedding history length.
                - normalise (bool, default=True).
        """
        self._measure = MeasureType.TE
        self._measure_title = 'Transfer Entropy'
        return self.__setup_mite_calc(estimator, include_intra, calc_statsig, stat_sig_perm_num, p_threshold, vis, plot_epochs, **kwargs)


    def compute_atoms(self, tau: int = 1, redundancy: str = 'MMI', include_intra: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to compute Integrated Information Decomposition (ΦID) between data (time-series signals) instantiated in the HyperIT object.

        Args:
            tau                 (int, optional): Time-lag parameter. Defaults to 1.
            redundancy          (str, optional): Redundancy function to use. Defaults to 'MMI' (Minimum Mutual Information).
            include_intra      (bool, optional): Whether to include intra-brain analysis. Defaults to False.
            
        Returns:
            np.ndarray: A matrix of integrated information decomposition atoms. The shape of the matrix is determined by
            the `include_intra` and `calc_statsig` flags:
                - If `include_intra` is False, shape is (n_epo, n_freq_bands, n_chan, n_chan, 16).
                - If `include_intra` is True, shape is (n_epo, n_freq_bands, 2*n_chan, 2*n_chan, 16).
                
        Note:
            When `include_intra` is True, the matrix can be segmented accordingly:
                - `intra1`: matrix[:, :, :n_chan, :n_chan]
                - `intra2`: matrix[:, :, n_chan:, n_chan:]
                - `inter12`: matrix[:, :, :n_chan, n_chan:]
                - `inter21`: matrix[:, :, n_chan:, :n_chan]
            Visualisation is not a possibility at the moment.

        Available Redundancy Functions:
            - 'MMI': Minimum Mutual Information
            - 'CCS': Common Change in Surprisal

        """
        
        self._measure = MeasureType.PhyID
        self._measure_title = 'Integrated Information Decomposition'
        return self.__setup_atom_calc(tau, redundancy, include_intra)
