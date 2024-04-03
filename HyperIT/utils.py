import numpy as np
from jpype import isJVMStarted, getDefaultJVMPath, startJVM, JArray, JDouble, shutdownJVM
import os

def setup_JVM(working_directory: str = None) -> None:
        if(not isJVMStarted()):
            
            if working_directory is None:
                working_directory = os.getcwd()

            jarLocation = os.path.join(working_directory, "infodynamics.jar")

            if not os.path.isfile(jarLocation):
                    raise FileNotFoundError(f"infodynamics.jar not found (expected at {os.path.abspath(jarLocation)}).")

            startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


def setup_JArray(a: np.ndarray) -> JArray:
    """ Converts a numpy array to a Java array for use in JIDT."""

    a = (a).astype(np.float64) 

    try:
        ja = JArray(JDouble, a.ndim)(a)
    except Exception: 
        ja = JArray(JDouble, a.ndim)(a.tolist())

    return ja


def convert_names_to_indices(_channel_names, roi_part, participant):
    """Converts ROI channel names or groups of names into indices based on the channel list.

    Args:
        roi_part: A single ROI, list of ROIs, or list of lists of ROIs to convert.
        participant: The index of the participant (0 or 1) to match with the correct channel names list.

    Returns:
        A list of indices, or list of lists of indices, corresponding to the channel names.
    """
    channel_names = _channel_names[participant]

    # Handle sub-sublists (grouped channel comparison)
    if all(isinstance(item, list) for item in roi_part):
        # roi_part is a list of lists
        return [[channel_names.index(name) if isinstance(name, str) else name for name in group] for group in roi_part]
    
    # Handle simple list (pointwise channel comparison)
    elif isinstance(roi_part, list):
        return [channel_names.index(name) if isinstance(name, str) else name for name in roi_part]

    # Handle single channel name or index
    elif isinstance(roi_part, str):
        return [channel_names.index(roi_part)]

    else:
        return roi_part  # In case roi_part is already in the desired format (indices)
    

def convert_indices_to_names(_channel_names, roi_part, participant):
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

