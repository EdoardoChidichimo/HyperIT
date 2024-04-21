import numpy as np
from jpype import JArray, JDouble, JPackage, JClass
import mne
import warnings

warnings.filterwarnings("ignore", message="filter_length .* is longer than the signal .*")

text_positions = {'rtr': (485, 1007),
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


def bandpass_filter_data(data: np.ndarray, sfreq: float, freq_bands: dict, **filter_options) -> np.ndarray:

    n_epochs, n_channels, n_samples = data.shape
    n_freq_bands = len(freq_bands)

    filtered_data = np.empty((n_epochs, n_freq_bands, n_channels, n_samples))

    for i, (band, (l_freq, h_freq)) in enumerate(freq_bands.items()):
        filtered_data[:,i,:,:] = mne.filter.filter_data(data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False, **filter_options)

    return filtered_data # returns with shape of (n_epochs, n_freq_bands, n_channels, n_samples)

def setup_JArray(a: np.ndarray) -> JArray:
    """ Converts a numpy array to a Java array for use in JIDT."""

    a = (a).astype(np.float64) 

    try:
        ja = JArray(JDouble, a.ndim)(a)
    except Exception: 
        ja = JArray(JDouble, a.ndim)(a.tolist())

    return ja


def convert_names_to_indices(_channel_names, roi_part, participant) -> list:
    """Converts ROI channel names or groups of names into indices based on the channel list.

    Args:
        roi_part: A single ROI, list of ROIs, or list of lists of ROIs to convert.
        participant: The index of the participant (0 or 1) to match with the correct channel names list.

    Returns:
        A list of indices, or list of lists of indices, corresponding to the channel names.
    """
    channel_names = _channel_names[participant]

    def get_index(name):
        if not isinstance(name, (str, int)):
            raise TypeError(f"Unsupported type {type(name)} in ROI specification.")
        if isinstance(name, str):
            try:
                return channel_names.index(name)
            except ValueError:
                raise ValueError(f"Channel name '{name}' not found in participant {participant}'s channel list.")
        return name
    
    if isinstance(roi_part, (list, tuple)):  # support tuples as well
        if all(isinstance(item, list) for item in roi_part):
            return [[get_index(name) for name in group] for group in roi_part]
        else:
            return [get_index(name) for name in roi_part]
    elif isinstance(roi_part, str):
        return [get_index(roi_part)]
    else:
        raise TypeError("ROI must be a list, tuple, or string of channel names/indices.")


def convert_indices_to_names(_channel_names, roi_part, participant) -> list:
    """Converts ROI channel indices or groups of indices into names based on the channel list.

    Args:
        roi_part: A single index, list of indices, or list of lists of indices to convert.
        participant: The index of the participant (0 or 1) to match with the correct channel names list.

    Returns:
        A list of names, or list of lists of names, corresponding to the channel indices.
    """
    channel_names = _channel_names[participant]
    
    if isinstance(roi_part, np.ndarray):
        roi_part = roi_part.tolist()

    # Handle sub-sublists (grouped channel comparison)
    if all(isinstance(item, list) for item in roi_part):
        return [[channel_names[index] if isinstance(index, int) else index for index in group] for group in roi_part]

    # Handle simple list (pointwise channel comparison)
    elif isinstance(roi_part, list):
        return [channel_names[index] if isinstance(index, int) else index for index in roi_part]

    # Handle single channel index
    elif isinstance(roi_part, int):
        return channel_names[roi_part]

    else:
        return roi_part  # In case roi_part is already in the desired format (names)


def set_estimator(estimator_type: str, measure: str, params: dict) -> tuple:
    
    estimator_config = {
        'mi': {
            'histogram': ('Histogram/Binning Estimator', None, {}),
            'ksg1': ('KSG Estimator (version 1)', 'infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1', {'k': str(params.get('kraskov_param', 4))}),
            'ksg2': ('KSG Estimator (version 2)', 'infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov2', {'k': str(params.get('kraskov_param', 4))}),
            'kernel': ('Box Kernel Estimator', 'infodynamics.measures.continuous.kernel.MutualInfoCalculatorMultiVariateKernel', {'KERNEL_WIDTH': str(params.get('kernel_width', 0.25))}),
            'gaussian': ('Gaussian Estimator', 'infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian', {}),
            'symbolic': ('Symbolic Estimator', None, {})
        },
        'te': {
            'ksg': ('KSG Estimator', 'infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorMultiVariateKraskov', {
                "k_HISTORY": str(params.get('k', 1)), "k_TAU": str(params.get('k_tau', 1)),
                "l_HISTORY": str(params.get('l', 1)), "l_TAU": str(params.get('l_tau', 1)),
                "DELAY": str(params.get('delay', 1)), "k": str(params.get('kraskov_param', 4))
            }),
            'kernel': ('Box Kernel Estimator', 'infodynamics.measures.continuous.kernel.TransferEntropyCalculatorMultiVariateKernel', {'KERNEL_WIDTH': str(params.get('kernel_width', 0.5))}),
            'gaussian': ('Gaussian Estimator', 'infodynamics.measures.continuous.gaussian.TransferEntropyCalculatorMultiVariateGaussian', {
                "k_HISTORY": str(params.get('k', 1)), "k_TAU": str(params.get('k_tau', 1)),
                "l_HISTORY": str(params.get('l', 1)), "l_TAU": str(params.get('l_tau', 1)),
                "DELAY": str(params.get('delay', 1)), "BIAS_CORRECTION": str(params.get('bias_correction', False)).lower()
            }),
            'symbolic': ('Symbolic Estimator', 'infodynamics.measures.continuous.symbolic.TransferEntropyCalculatorSymbolic', {"k_HISTORY": str(params.get('k', 1))}, 2)
        }
    }

    estimator_name, calculator_path, properties, initialise_parameter = None, None, {}, None
    measure_config = estimator_config.get(measure.lower(), {})
    config = measure_config.get(estimator_type.lower(), None)

    if config:
        estimator_name, calculator_path, properties = config[:3]
        initialise_parameter = config[3] if len(config) > 3 else None
        if calculator_path:
            
            try:
                calculator = JClass(calculator_path)

            except Exception as e:
                raise RuntimeError(f"Failed to load and instantiate the calculator: {e}")
        else:
            calculator = None

        if measure.lower() == 'te' or estimator_type.lower() not in ['histogram', 'symbolic']:
            properties['NORMALISE'] = 'true' if params.get('normalise', True) else 'false'
    else:
        raise ValueError(f"Estimator type {estimator_type} not supported for measure {measure}.")

    return estimator_name, calculator, properties, initialise_parameter
