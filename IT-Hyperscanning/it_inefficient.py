import numpy as np
from scipy import stats
from scipy.stats import mode
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from jpype import *


## SETUP
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


## DATA-TYPE CHECKS
def setup_JArray(a: np.ndarray) -> JArray:
    
    a = (a).astype(np.float64) 

    try:
        ja = JArray(JDouble, 1)(a)
        
    except Exception: 
        ja = JArray(JDouble, 1)(a.tolist())

    return ja
        
def check_eeg_data(data: np.ndarray, is_epoched: bool) -> np.ndarray:

    if not isinstance(data, np.ndarray):
        raise ValueError("EEG data must be a numpy array.")
    
    assert data[0].shape[0] == data[1].shape[0], "Data passed should have the same number of epochs for each participant."

    if data.ndim == 3 and not is_epoched:
        raise ValueError(f"is_epoched is set to False but EEG signals passed have {data.ndim} dimensions. For unepoched data, expected 2 dimensions with shape (n_channels, n_timepoints); for epoched data, expected 3 dimensions with shape (n_epochs, n_channels, n_samples)")
    elif data.ndim > 3:
        raise ValueError(f"The EEG signals passed do not have the correct shape. Expected 2 dimensions (n_chan, time_points) or 3 dimensions (n_epochs, n_chan, time_points); instead, received {data.ndim}.")
    return data

def calc_fd_bins(X: np.array, Y: np.array) -> int:
    # Freedman-Diaconis Rule for frequency-distribution bin size
    fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
    fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
    fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
    return fd_bins


## MUTUAL INFORMATION 

#Add permutation testing to find distr_mean, distr_std, and pval
def mi_hist(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, bins: int) -> float:
    """Calculate mutual information between two time series using histogram/binning approach (with Freedman-Diaconis rule to determine bin size)

    Args:
        s1                  (np.ndarray): EEG time series 1
        s2                  (np.ndarray): EEG time series 2
        is_epoched                (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        bins             (int, optional): Number of bins to discretise/bin the signals. Defaults to None and uses Freedman-Diaconis rule to compute optimum.

    Returns:
                                 (float): Mutual information binning/histogram estimation 
    """

    n_epo = s1.shape[0] if is_epoched else 1

    entropy_X = 0
    entropy_Y = 0
    entropy_XY = 0

    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

        # Bin edges
        edges_X = np.linspace(min(X), max(X), bins)
        edges_Y = np.linspace(min(Y), max(Y), bins)
        
        # What bin each value should be in
        X_indices = np.digitize(X, edges_X, right=False)
        Y_indices = np.digitize(Y, edges_Y, right=False)

        counts_X = np.bincount(X_indices, minlength=bins)[1:]
        counts_Y = np.bincount(Y_indices, minlength=bins)[1:]

        pmd_X = counts_X.astype(float) / len(X)
        pmd_Y = counts_Y.astype(float) / len(Y)

        # For each value, an array of bin length, with 1 indicating that bin (and 0 not)
        mask_X = (X_indices[:, None] == np.arange(1, bins + 1)).astype(int)
        mask_Y = (Y_indices[:, None] == np.arange(1, bins + 1)).astype(int)

        freq = np.dot(mask_X.T, mask_Y).astype(float)
        jpmd = freq / np.sum(freq)

        entropy_X += -np.sum(pmd_X * np.log2(pmd_X + np.finfo(float).eps)) 
        entropy_Y += -np.sum(pmd_Y * np.log2(pmd_Y + np.finfo(float).eps))
        entropy_XY += -np.sum(jpmd * np.log2(jpmd + np.finfo(float).eps))

    three_H = np.array([entropy_X / n_epo, entropy_Y / n_epo, entropy_XY / n_epo])
    mi = three_H[0] + three_H[1] - three_H[2]

    return mi
    
def mi_ksg(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, version: int = 1, kraskov_param: int = 4, stat_sig_perm_num: int = 100) -> Tuple[float, np.ndarray]:
    """Calculate mutual information between two time series using Kraskov-Stögbauer-Grassberger estimator 1 or 2

    Args:
        s1                  (np.ndarray): EEG time series 1
        s2                  (np.ndarray): EEG time series 2
        is_epoched                (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        version          (int, optional): Which estimator version to use. Defaults to 1.
        kraskov_param    (int, optional): KSG parameter. Defaults to 4.

    Returns:
                                 (float): KSG mutual information between s1 and s2.
    """
    n_epo = s1.shape[0] if is_epoched else 1
    mi = np.zeros((n_epo, 4))

    miCalcClass = JPackage("infodynamics.measures.continuous.kraskov")\
                  .MutualInfoCalculatorMultiVariateKraskov1 if version == 1 else \
                  JPackage("infodynamics.measures.continuous.kraskov")\
                  .MutualInfoCalculatorMultiVariateKraskov2
    
    miCalc = miCalcClass()
    miCalc.setProperty("k", str(kraskov_param))

    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)

        miCalc.initialise()
        miCalc.setObservations(sig1, sig2)

        result = miCalc.computeAverageLocalOfObservations() * np.log(2)
        if calc_sigstats:
            stat_sig = miCalc.computeSignificance(stat_sig_perm_num)
            distr_mean = stat_sig.getMeanOfDistribution()
            distr_std = stat_sig.getStdOfDistribution()
            p_val = stat_sig.pValue

            mi[epo_i] = [result, distr_mean, distr_std, p_val]
        else:
            mi[epo_i] = result

    average_result = np.mean(mi[:, 0])

    return average_result, mi

def mi_kernel(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, kernel_width: float = 0.25, stat_sig_perm_num: int = 100) -> Tuple[float, np.ndarray]:
    """Calculate mutual information between two time series using a kernel estimator
    
    Args:
        s1                  (np.ndarray): EEG time series 1
        s2                  (np.ndarray): EEG time series 2
        is_epoched                (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        kernel_width   (float, optional): Kernel width to use. Defaults to 0.25.

    Returns:
                                 (float): Kernel mutual information between s1 and s2.
    """
    n_epo = s1.shape[0] if is_epoched else 1
    mi = np.zeros((n_epo, 4))

    miCalcClass = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
    miCalc = miCalcClass()
    miCalc.setProperty("KERNEL_WIDTH", str(kernel_width))

    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)

        miCalc.initialise()
        miCalc.setObservations(sig1, sig2)
        
        result = miCalc.computeAverageLocalOfObservations() * np.log(2)

        if calc_sigstats:
            stat_sig = miCalc.computeSignificance(stat_sig_perm_num)
            distr_mean = stat_sig.getMeanOfDistribution()
            distr_std = stat_sig.getStdOfDistribution()
            p_val = stat_sig.pValue

            mi[epo_i] = [result, distr_mean, distr_std, p_val]
        
        else:
            mi[epo_i] = result

    average_result = np.mean(mi[:, 0])

    return average_result, mi

def mi_gaussian(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, stat_sig_perm_num: int = 100) -> np.ndarray:
    """Calculate mutual information between two time series using Gaussian estimator
    
    Args:
        s1            (np.ndarray): EEG time series 1
        s2            (np.ndarray): EEG time series 2
        is_epoched          (bool): Whether the data is epoched. If True, s1 and s2 take shape (n_epo, n_chan, n_samples); if False, take shape (n_chan, n_samples).

    Returns:
                           (float): Gaussian mutual information between s1 and s2.
    """


    n_epo = s1.shape[0] if is_epoched else 1
    mi = np.zeros((n_epo, 4))

    miCalcClass = JPackage("infodynamics.measures.continuous.gaussian").MutualInfoCalculatorMultiVariateGaussian
    miCalc = miCalcClass()

    if is_epoched:
        X, Y = JArray(JDouble, 2)(s1.T), JArray(JDouble, 2)(s2.T) # shape: ([n_samples, n_chan])
    else:
        X, Y = JArray(JDouble, 1)(s1), JArray(JDouble, 1)(s2) # ([n_samples])
    
    miCalc.initialise(n_epo, n_epo)
    miCalc.setObservations(X, Y)
    result = miCalc.computeAverageLocalOfObservations() * np.log(2)

    if calc_sigstats:
        stat_sig = miCalc.computeSignificance(stat_sig_perm_num)
        distr_mean = stat_sig.getMeanOfDistribution()
        distr_std = stat_sig.getStdOfDistribution()
        p_val = stat_sig.pValue
        return result, np.array((result, distr_mean, distr_std, p_val))

    return result, None

def entropy_symb(s1: np.ndarray, s2: np.ndarray, is_epoched: bool, l: int, m: int) -> np.ndarray:
    """Calculate 3H symbolic Shannon entropic measures of two given time series signals

    Args:
        s1            (np.ndarray): EEG signal 1
        s2            (np.ndarray): EEG signal 2
        is_epoched          (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        l                    (int): Lag step
        m                    (int): Embedding dimension

    Returns:
        np.ndarray: An array of Shannon entropic measures ([H(X), H(Y), H(X,Y)])
    """

    def symb_symbolise(X: np.ndarray, l: int, m: int):
        Y = np.empty((m, len(X) - (m - 1) * l))
        for i in range(m):
            Y[i] = X[i * l:i * l + Y.shape[1]]
        return Y.T
        
    def symb_incr_counts(key,d):
        d[key] = d.get(key, 0) + 1

    def symb_normalise(d):
        s=sum(d.values())        
        for key in d:
            d[key] /= s



    n_epo = s1.shape[0] if is_epoched else 1
    entropy_X, entropy_Y, entropy_XY = 0, 0, 0

    hashmult = np.power(m, np.arange(m))
    
    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

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
        
        entropy_X += -np.sum(p_x * np.log2(p_x + np.finfo(float).eps)) 
        entropy_Y += -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
        entropy_XY += -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))

    # If multiple epochs, calculates average entropy of a channel across epochs by summating entropy at each epoch and dividing by total no. of epochs. 
    # If incoming signals are already averaged across epochs (and therefore n_epo = 1), returns the entropy of that channel.
    avg_epo_entropies = np.array([entropy_X / n_epo, entropy_Y / n_epo, entropy_XY / n_epo])

    return avg_epo_entropies
#Add permutation testing to find distr_mean, distr_std, and pval
def mi_symbolic(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, l: int = 1, m: int = 3) -> np.ndarray:
    """Calculate symbolic mutual information between two time series using 3H method

    Args:
        s1              (np.ndarray): EEG time series 1
        s2              (np.ndarray): EEG time series 2
        is_epoched            (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        l            (int, optional): Lag step. Defaults to 1.
        m            (int, optional): Embedding dimension. Defaults to 3.

    Returns:
                             (float): Symbolic mutual information between s1 and s2.
    """
    entropies = entropy_symb(s1, s2, is_epoched, l, m) # entropy_symb returns [n_ch, 3] for H(X), H(Y), and H(X,Y)
    mi = entropies[0] + entropies[1] - entropies[2]
    return mi


def compute_mi(eeg_1: np.ndarray, eeg_2: np.ndarray = None, calc_sigstats: bool = False, mode: str = "kernel",  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Main function to compute mutual information between all EEG channel combinations, whether intra- or inter-brain. 
       Incoming data can be epoched or already epoch-averaged. 
       Different estimators (mode) available:
        
         -  Binning/Histogram
         -  Kraskov-Stögbauer-Grassberger (KSG; 1 or 2)
         -  Kernel 
         -  Gaussian
         -  Symbolic

    Args:
        eeg_1                     (np.ndarray): Participant (1) EEG data
        eeg_2           (np.ndarray, optional): Participant 2 EEG data . Defaults to None.
        is_epoched                      (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        mode                   (str, optional): Which estimator type to compute MI with. Defaults to "kernel".

    Returns:
        mi_matrix                 (np.ndarray): A mutual information matrix of all channel combinations. Note that the intra-brain MI will be symmetric.
    """


    inter_brain = eeg_2 is not None
    is_epoched = eeg_1.ndim == 3 or (eeg_1.ndim == 3 and eeg_2.ndim == 3) 

    signal1 = check_eeg_data(eeg_1, is_epoched)
    signal2 = check_eeg_data(eeg_2, is_epoched) if inter_brain else signal1

    n_chan = signal1.shape[1 if is_epoched else 0]
    n_epo = signal1.shape[0] if is_epoched else 1

    mi_estimation_methods = {
        "hist": mi_hist,
        "ksg": mi_ksg,
        "kernel": mi_kernel,
        "gaussian": mi_gaussian,
        "symbolic": mi_symbolic
    }

    if mode not in mi_estimation_methods:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes are: {list(mi_estimation_methods.keys())}.")
    
    if mode == "hist" and kwargs.get('bins') is None:
        fd_bins_tot = 0
        tot = 0
        for chan_i in range(n_chan):
            for chan_j in range(chan_i, n_chan):
                x, y = (signal1[:, chan_i, :], signal2[:, chan_j, :]) if is_epoched else (signal1[chan_i, :], signal2[chan_j, :])
                tot += 1
                fd_bins_tot += calc_fd_bins(x, y)

        kwargs['bins'] = round(fd_bins_tot / tot)
        print("Optimum number of frequency-distribution bins with which to discretise signals, as given by Freedman-Diaconis rule, is",kwargs['bins'])

    mi_func = mi_estimation_methods[mode]

    mi_matrix = np.zeros((n_chan, n_chan))
    sigstats = np.zeros((n_chan, n_chan, n_epo, 4))

    for i in tqdm(range(n_chan)):
        start_j = 0 if inter_brain else i
        for j in range(start_j, n_chan):
            if inter_brain or i != j:
                s1, s2 = (signal1[:, i, :], signal2[:, j, :]) if is_epoched else (signal1[i, :], signal2[j, :]) # whether to keep epochs
                
                try:
                    mi_matrix[i, j], sigstats[i,j] = mi_func(s1, s2, calc_sigstats, is_epoched, **kwargs)
                except TypeError:
                    mi_matrix[i, j] = mi_func(s1, s2, calc_sigstats, is_epoched, **kwargs)
                    sigstats[i, j] = None 
                
                if not inter_brain:
                    try:
                        mi_matrix[j, i], sigstats[j,i] = mi_matrix[i, j], sigstats[i,j] # or 0 if you want to avoid symmetry    
                    except TypeError:
                        mi_matrix[j, i], sigstats[j,i] = mi_matrix[i, j], None
                        sigstats[j,i] = None
                    
    return mi_matrix, sigstats



## TRANSFER ENTROPY
def te_ksg(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, optimise: bool = False, k: int = 1, k_tau: int = 1, l: int = 1, l_tau: int = 1, delay: int = 1, kraskov_param: int = 4, stat_sig_perm_num: int = 100) -> Tuple[float, np.ndarray]:
    """Calculates transfer entropy between 2 time series using Kraskov-Stögbauer-Grassberger (KSG) Estimator 

    Args:
        s1                (np.ndarray): EEG time series 1 (SOURCE)
        s2                (np.ndarray): EEG time series 2 (TARGET)
        is_epoched              (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        optimise                (bool): Whether to find and use optimal k, k_tau, l, l_tau, and delay parameters. Defaults to False.
        k              (int, optional): TARGET history embedding length (i.e., length of past to consider). Defaults to 1.
        k_tau          (int, optional): TARGET history embedding delay (i.e., applied between elements of embedding vector). Defaults to 1.
        l              (int, optional): SOURCE history embedding length. Defaults to 1.
        l_tau          (int, optional): SOURCE history embedding delay. Defaults to 1.
        delay          (int, optional): Delay from SOURCE to TARGET. Defaults to 1.
        kraskov_param  (int, optional): Kraskov parameter for number of nearest searches. Defaults to 4.

    Returns:
                               (float): KSG TE estimation (s1->s2)
    """

    def te_kraskov_find_optimal_parameters(eeg_1: np.ndarray, eeg_2: np.ndarray = None, is_epoched: bool = True) -> np.ndarray:
        """Find the optimal k, k_tau, l, l_tau, and delay values for Kraskov Transfor Entropy estimation using the Ragwitz method

        Args:
            eeg1                      (np.ndarray): EEG time series 1 (SOURCE).
            eeg2            (np.ndarray, optional): EEG time series 2 (TARGET). Defaults to None.
            is_epoched                      (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)


        Returns:
                                        (np.array): List of optimal k, k_tau, l, l_tau, and delay values based on their respective statistical mode
        """
        
        # Do not need all epochs, so take first epoch if eeg_1 has not been epoch averaged. 
        sig1 = check_eeg_data(eeg_1, is_epoched)[0] if is_epoched else check_eeg_data(eeg_1, is_epoched)
        n_chan = sig1.shape[0]
        inter_brain = eeg_2 is not None

        if inter_brain:
            sig2 = check_eeg_data(eeg_2, is_epoched)[0] if is_epoched else check_eeg_data(eeg_2, is_epoched)
        else:
            sig2 = sig1

        K_values, KTau_values, L_values, LTau_values, delay_values = [], [], [], [], []
        teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        teCalc = teCalcClass()
        teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
                        teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
        teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "6")
        teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "6")

        for i in tqdm(range(n_chan)):
            for j in range(n_chan):
                if inter_brain or i != j:

                    X, Y = sig1[i, :], sig2[j, :]
                    s1 = setup_JArray(X)
                    s2 = setup_JArray(Y)

                    teCalc.initialise()
                    teCalc.setObservations(s1, s2)

                    K_values.append(int(str(teCalc.getProperty(teCalcClass.K_PROP_NAME))))
                    KTau_values.append(int(str(teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME))))
                    L_values.append(int(str(teCalc.getProperty(teCalcClass.L_PROP_NAME))))
                    LTau_values.append(int(str(teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME))))
                    delay_values.append(int(str(teCalc.getProperty(teCalcClass.DELAY_PROP_NAME))))


        def get_mode_value(modes_result):
            if np.isscalar(modes_result.mode):
                return modes_result.mode
            else:
                return modes_result.mode[0]

        return np.array((get_mode_value(mode(K_values)),
                        get_mode_value(mode(KTau_values)),
                        get_mode_value(mode(L_values)),
                        get_mode_value(mode(LTau_values)),
                        get_mode_value(mode(delay_values)))) 


    if optimise:
        k, k_tau, l, l_tau, delay = te_kraskov_find_optimal_parameters(s1, s2, is_epoched)

    n_epo = s1.shape[0] if is_epoched else 1
    te = np.zeros((n_epo, 4))

    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.setProperty("k_HISTORY", str(k))
    teCalc.setProperty("k_TAU", str(k_tau))
    teCalc.setProperty("l_HISTORY", str(l))
    teCalc.setProperty("l_TAU", str(l_tau))
    teCalc.setProperty("DELAY", str(delay))
    teCalc.setProperty("k", str(kraskov_param))

    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)
        
        teCalc.initialise() 
        teCalc.setObservations(sig1, sig2)

        result = teCalc.computeAverageLocalOfObservations() * np.log(2)

        stat_sig = teCalc.computeSignificance(stat_sig_perm_num)
        distr_mean = stat_sig.getMeanOfDistribution()
        distr_std = stat_sig.getStdOfDistribution()
        p_val = stat_sig.pValue

        te[epo_i] = [result, distr_mean, distr_std, p_val]

    average_result = np.mean(te[:, 0])

    return average_result, te

def te_kernel(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, k: int = 1, kernel_width: float = 0.5, stat_sig_perm_num: int = 100) -> Tuple[float, np.ndarray]:
    """Calculates transfer entropy between 2 time series using kernel estimator

    Args:
        s1                   (np.ndarray): EEG time series 1 (SOURCE)
        s2                   (np.ndarray): EEG time series 2 (TARGET)
        is_epoched                 (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        k                 (int, optional): TARGET history embedding length, see Schreiber (2000). Defaults to 1. 
        kernel_width    (float, optional): Kernel width of normalised units. Defaults to 0.5.

    Returns:
                                  (float): Kernel TE estimation (s1->s2) 
    """

    n_epo = s1.shape[0] if is_epoched else 1
    te = np.zeros((n_epo, 4))

    teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true") 

    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)
    
        teCalc.initialise(k, kernel_width) 
        teCalc.setObservations(sig1, sig2)

        result = teCalc.computeAverageLocalOfObservations() * np.log(2)

        stat_sig = teCalc.computeSignificance(stat_sig_perm_num)
        distr_mean = stat_sig.getMeanOfDistribution()
        distr_std = stat_sig.getStdOfDistribution()
        p_val = stat_sig.pValue

        te[epo_i] = [result, distr_mean, distr_std, p_val]

    average_result = np.mean(te[:, 0])

    return average_result, te

def te_gaussian(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, k: int = 1, k_tau: int = 1, l: int = 1, l_tau: int = 1, delay: int = 1, bias_correction: bool = False, stat_sig_perm_num: int = 100) -> Tuple[float, np.ndarray]:
    """Calculates transfer entropy between 2 time series using Gaussian estimator

    Args:
        s1                     (np.ndarray): EEG time series 1 (SOURCE)
        s2                     (np.ndarray): EEG time series 2 (TARGET)
        is_epoched                   (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        k                   (int, optional): TARGET history embedding length (i.e., length of past to consider). Defaults to 1.
        k_tau               (int, optional): TARGET history embedding delay (i.e., applied between elements of embedding vector). Defaults to 1.
        l                   (int, optional): SOURCE history embedding length. Defaults to 1.
        l_tau               (int, optional): SOURCE history embedding delay. Defaults to 1.
        delay               (int, optional): Delay from SOURCE to TARGET. Defaults to 1.
        bias_correction    (bool, optional): _description_. Defaults to False.

    Returns:
                                    (float): Gaussian TE estimation (s1->s2)
    """
    
    n_epo = s1.shape[0] if is_epoched else 1
    te = np.zeros((n_epo, 4))

    teCalcClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
    teCalc = teCalcClass()
    teCalc.setProperty("k_HISTORY", str(k))
    teCalc.setProperty("k_TAU", str(k_tau))
    teCalc.setProperty("l_HISTORY", str(l))
    teCalc.setProperty("l_TAU", str(l_tau))
    teCalc.setProperty("DELAY", str(delay))
    teCalc.setProperty("BIAS_CORRECTION", str(bias_correction).lower())

    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)

        teCalc.initialise()
        teCalc.setObservations(sig1, sig2)
        
        result = teCalc.computeAverageLocalOfObservations() * np.log(2)

        stat_sig = teCalc.computeSignificance(stat_sig_perm_num)
        distr_mean = stat_sig.getMeanOfDistribution()
        distr_std = stat_sig.getStdOfDistribution()
        p_val = stat_sig.pValue

        te[epo_i] = [result, distr_mean, distr_std, p_val]

    average_result = np.mean(te[:, 0])

    return average_result, te

def te_symbolic(s1: np.ndarray, s2: np.ndarray, calc_sigstats: bool, is_epoched: bool, k: int = 1, stat_sig_perm_num: int = 100) -> np.ndarray:
    """Calculates Transfer Entropy between 2 Univariate Time Series using Symbolic Estimation (Staniek & Lehrnertz)

    Args:
        s1             (np.ndarray): EEG time series 1 (SOURCE)
        s2             (np.ndarray): EEG time series 2 (TARGET)
        is_epoched           (bool): Whether the data is epoched. If True, data takes shape (n_epo, n_chan, n_samples); if False, data takes shape (n_chan, n_samples)
        k           (int, optional): TARGET history embedding length. Defaults to 1.

    Returns:
        result       (float): Symbolic TE estimation (s1->s2)
    """
    n_epo = s1.shape[0] if is_epoched else 1
    te = np.zeros((n_epo, 4))

    teCalcClass = JPackage("infodynamics.measures.continuous.symbolic").TransferEntropyCalculatorSymbolic
    teCalc = teCalcClass()
    teCalc.setProperty("k_HISTORY", str(k))

    for epo_i in range(n_epo):

        X, Y = (s1[epo_i, :], s2[epo_i, :]) if is_epoched else (s1, s2)

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)

        teCalc.initialise(2) # base = 2
        teCalc.setObservations(sig1, sig2)
        
        result = teCalc.computeAverageLocalOfObservations() * np.log(2)

        stat_sig = teCalc.computeSignificance(stat_sig_perm_num)
        distr_mean = stat_sig.getMeanOfDistribution()
        distr_std = stat_sig.getStdOfDistribution()
        p_val = stat_sig.pValue

        te[epo_i] = [result, distr_mean, distr_std, p_val]

    average_result = np.mean(te[:, 0])

    return average_result, te


def compute_te(eeg_1: np.ndarray, eeg_2: np.ndarray = None, calc_sigstats: bool = False, mode: str = "kernel", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Main function to compute transfer entropy between all EEG channel combinations, whether intra- or inter-brain. 
       Incoming data can be epoched or already epoch-averaged. 
       Different estimators (mode) available:
        
         -  Kraskov-Stögbauer-Grassberger (KSG)
         -  Kernel 
         -  Gaussian
         -  Symbolic
         
    Args:
        eeg_1                     (np.ndarray): Participant (1) EEG data with shape: either (n_epo, n_chan, n_samples) or (n_chan, n_samples)
        eeg_2           (np.ndarray, optional): Participant 2 EEG data. Defaults to None.
        mode                   (str, optional): Which estimator type to compute MI with. Defaults to "ksg".

    Returns:
        te_matrix                 (np.ndarray): Two transfer entropy matrices of all channel combinations (eeg_1 -> eeg_2, eeg_2 -> eeg_1) for inter-brain, and a single matrix for intra-brain analysis.
    """

    inter_brain = eeg_2 is not None
    is_epoched = eeg_1.ndim == 3 or (eeg_1.ndim == 3 and eeg_2.ndim == 3) 

    signal1 = check_eeg_data(eeg_1, is_epoched)
    signal2 = check_eeg_data(eeg_2, is_epoched) if inter_brain else signal1

    n_chan = signal1.shape[1 if is_epoched else 0]
    n_epo = signal1.shape[0] if is_epoched else 1

    te_estimation_methods = {
        "ksg": te_ksg,
        "kernel": te_kernel,
        "gaussian": te_gaussian,
        "symbolic": te_symbolic
    }

    if mode not in te_estimation_methods:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes are: {list(te_estimation_methods.keys())}.")
    
    te_func = te_estimation_methods[mode]

    te_matrix_xy = np.zeros((n_chan, n_chan))
    te_matrix_yx = np.zeros((n_chan, n_chan))
    sigstats_xy = np.zeros((n_chan, n_chan, n_epo, 4))
    sigstats_yx = np.zeros((n_chan, n_chan, n_epo, 4))

    for i in tqdm(range(n_chan)):
        for j in range(n_chan):
            if inter_brain or i != j: # avoid self-channel calculations for intra_brain condition
                s1, s2 = (signal1[:, i, :], signal2[:, j, :]) if is_epoched else (signal1[i, :], signal2[j, :]) # whether to keep epochs
            
                try:
                    te_matrix_xy[i, j], sigstats_xy[i, j] = te_func(s1, s2, calc_sigstats, is_epoched, **kwargs)
                except TypeError:
                    te_matrix_xy[i, j] = te_func(s1, s2, calc_sigstats, is_epoched, **kwargs)
                    sigstats_xy[i, j] = None 

                if inter_brain: # don't need to compute opposite matrix for intra-brain as we already loop through each channel combination including symmetric
                    try:
                        te_matrix_yx[i, j], sigstats_yx[i, j] = te_func(s2, s1, calc_sigstats, is_epoched, **kwargs)
                    except TypeError:
                        te_matrix_yx[i, j] = te_func(s2, s1, calc_sigstats, is_epoched, **kwargs)
                        sigstats_yx[i, j] = None 
                    
    if inter_brain:
        return te_matrix_xy, te_matrix_yx, sigstats_xy, sigstats_yx
    else:
        return te_matrix_xy, sigstats_xy


# def compute_complete_te(eeg_1: np.ndarray, eeg_2: np.ndarray = None, is_epoched: bool = True, mode: str = "ksg", **kwargs) -> np.ndarray:
#     """Computes complete transfer entropy by calculating conditional transfer entropy between two EEG signals conditioned on all other EEG signals.

#     Args:
#         eeg_1 (np.ndarray): _description_
#         eeg_2 (np.ndarray, optional): _description_. Defaults to None.
#         is_epoched (bool, optional): _description_. Defaults to True.
#         mode (str, optional): _description_. Defaults to "kernel".

#     Returns:
#         np.ndarray: _description_
#     """

#     inter_brain = eeg_2 is not None

#     signal1 = check_eeg_data(eeg_1, is_epoched)
#     signal2 = check_eeg_data(eeg_2, is_epoched) if inter_brain else signal1

#     n_chan = signal1.shape[1 if is_epoched else 0]
#     ch_axis = 1 if is_epoched else 0 # which axis represents channels

#     cte_estimation_methods = {
#         "hist": cte_hist,
#         "ksg": cte_ksg,
#         "gaussian": cte_gaussian,
#     }

#     if mode not in cte_estimation_methods:
#         raise ValueError(f"Unsupported mode '{mode}'. Supported modes are: {list(cte_estimation_methods.keys())}.")
    
#     te_func = cte_estimation_methods[mode]

#     cte_matrix_xy = np.zeros((n_chan, n_chan))

#     for i in range(n_chan):

#         if inter_brain:
#             non_i = np.delete(signal1, i, axis=ch_axis) # if epoched, (epo, ch w/out i, s); if not, (ch w/out i, s)

#         for j in range(n_chan):

#             if inter_brain or i != j:
#                 # remember that s2 = s1 for INTRA_brain
#                 s1, s2 = (signal1[:, i, :], signal2[:, j, :]) if is_epoched else (signal1[i, :], signal2[j, :])
                
#                 if inter_brain:
#                     # Can put non_i definition here but not dependent on j-for-loop so left outside to save redundant computation
#                     non_j = np.delete(signal2, j, axis=ch_axis)
#                     s3 = np.concatenate((non_i, non_j), axis=ch_axis) # if epoched, (epo, 2x(n_chan-1),s); if not, (2x(n_chan-1), s)

#                 else:
#                     # has all non i and j channels of EEG data. s3: epoched (n_epo, n_chan w/out i or j, n_s) or not epoched (n_chan-2 w/out i or j, n_s)
#                     s3 = np.delete(signal1, [i, j], axis=ch_axis)

#                 cte_matrix_xy[i, j] = te_func(s1, s2, s3, is_epoched, **kwargs)


#     return cte_matrix_xy
                
if __name__ == "__main__":
    setup_JIDT(os.getcwd())


def compute_PhiID_atoms():
    ## Emailed Ceballos about code.
    pass








    #print("CTE(X -> Y | Z) = %.4f nats (null: %.4f +/- %.4f std dev.; p(surrogate > measured)=%.5f from %d surrogates)" %\
     #       (result, stat_sig.getMeanOfDistribution(), stat_sig.getStdOfDistribution(), stat_sig.pValue, stat_sig_perm_num))

## VISUALISATION

def plot_it(it_matrix: np.ndarray, sigstats: np.ndarray, inter_brain: bool, channel_names: List[str]):
    """Plots heatmap of mutual information or transfer entropy values for either intra-brain or inter-brain design.

    Args:
        it_matrix (np.ndarray): Matrix with shape (n_chan1, n_chan2). Can be same channels (intra-brain) or two-person channel (inter-brain). Note that intrabrain MI will be a symmetric heatmap. 
        inter_brain (bool): Whether the analysis is inter-brain analysis or not (i.e., intra-brain)
        channel_names (List[str]): List of channel names of EEG signals. Either a single list for intra-brain or two lists for inter-brain. 
    """

    p_threshold = 0.05
    print(it_matrix.shape)

    if channel_names.ndim == 1 and not inter_brain:
        channel_names = [channel_names, channel_names]

    if sigstats is not None:
        n_epo = sigstats.shape[2]
    else:
        n_epo = 1

    unepoched = not(n_epo > 1)
    title = 'Inter-Brain' if inter_brain else 'Intra-Brain'

    ## check if sigstats is just a single value or a matrix (i.e., un-epoched or epochs maintained). If matrix, ask if to plot for every epoch or specific epochs.
    if not unepoched: #i.e., if epoched
        avg_across_epochs = False
        print(f"{n_epo} epochs detected. Do you want to plot for all epochs or specific epochs?")
        print("1. All epochs")
        print("2. Specific epoch")
        print("3. Average MI/TE across epochs")
        choice = input("Enter choice: ")
        if choice == "1":
            print("Plotting for all epochs.")
            epochs = range(n_epo)
        elif choice == "2":
            print(f"Choose from 1 to {sigstats.shape[2]}")
            epo_choice = input("Enter epoch number(s): ")
            selected_epochs = np.array([int(num.strip()) for num in epo_choice.replace(',', ' ').split()])
            try:
                epochs = [int(epo-1) for epo in selected_epochs]
            except ValueError:
                print("All epochs must be integer numbers. Please try again.")
        elif choice == "3":
            print("Plotting for average MI/TE across epochs. Note that p-values will not be shown.")
            avg_across_epochs = True
            epochs = [0]

        else:
            raise ValueError("Invalid choice. Please enter 1 or 2.")
        
    else:
        print("Un-epoched data detected. Plotting for un-epoched data.")
        avg_across_epochs = False
        epochs = [0]

    for epo_i in epochs:

        plt.figure(figsize=(16, 16))

        if avg_across_epochs or unepoched:
            plt.matshow(it_matrix, cmap='BuPu', vmin=0)
            plt.title(title, pad=20)
            ## CANNOT SHOW P-VALUES FOR OPTION 3 (AVERAGED ACROSS EPOCHS)
        else:
            plt.matshow(sigstats[:,:,epo_i,0], cmap='BuPu', vmin=0)
            plt.title(f'{title}; Epoch {epo_i+1}', pad=20)
            
        if not avg_across_epochs and sigstats is not None:
            for i in range(sigstats.shape[0]):
                for j in range(sigstats.shape[1]):
                    p_val = float(sigstats[i, j, epo_i, 3])
                    if p_val < p_threshold and (not inter_brain and i != j):
                        plt.text(j, i, f'p={p_val:.2f}', ha='center', va='center', color='white', fontsize=8, fontweight='bold')

        plt.colorbar()
        plt.xlabel('Target Channels')
        plt.ylabel('Source Channels')
        plt.xticks(range(it_matrix.shape[0]), channel_names[1], rotation=90) 
        plt.yticks(range(it_matrix.shape[0]), channel_names[0])
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
        plt.show()

        highest = np.max(it_matrix)
        channel_pair_with_highest = np.unravel_index(np.argmax(it_matrix), it_matrix.shape)
        print(f"Strongest regions: (Source Channel {channel_names[0][channel_pair_with_highest[0]]} --> " +
                                f" Target Channel {channel_names[1][channel_pair_with_highest[1]]}) = {highest}")
        

