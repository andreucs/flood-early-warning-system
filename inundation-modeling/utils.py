from ground_truth_measurement import GroundTruthMeasurement
from typing import List
import numpy as np

def river_with_length(length: int) -> np.ma.MaskedArray:
    """
    Generate a binary river mask of given half-width with random noise.

    Parameters
    ----------
    length : int
        Half-width of the river channel centered at column 64.

    Returns
    -------
    numpy.ma.MaskedArray
        Boolean mask where True indicates river presence.
    """
    # Create straight channel
    first_row = np.zeros(128, dtype=bool)
    first_row[64 - length: 64 + length] = True
    imap = np.tile(first_row, (128, 1)).astype(bool)

    # Add random noise for variability
    random_mask = np.random.rand(128, 128) > 0.95
    imap ^= random_mask

    return np.ma.masked_array(imap)


def generate_dummy_DEM(width: int = 128, height: int = 128) -> np.ndarray:
    """
    Generate a synthetic digital elevation model (DEM) array.

    The DEM is constructed by:
      1. Creating a symmetric gradient in the x-direction, offset by +100.
      2. Subtracting a gradient in the y-direction that decreases from 10 to 0.
      3. Creating a depression of -10 elevation in a subregion [y 32:64, x 100:120].

    Parameters
    ----------
    width : int, optional
        Number of columns in the DEM. Defaults to 128.
    height : int, optional
        Number of rows in the DEM. Defaults to 128.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (height, width) representing elevation.
    """
    # Base symmetric elevation
    x = np.arange(-16, 16, step=32 / width)
    base = np.tile(np.abs(x), (height, 1)) + 100

    # Subtract y-gradient from 10 down to 0
    y_grad = np.linspace(10, 0, num=height)
    dem = base - np.tile(y_grad, (width, 1)).T

    # Create a depression in a central window
    dem[32:64, 100:120] -= 10

    return dem


def generate_dummy_train_data() -> List[GroundTruthMeasurement]:
    """
    Generate a ground truth dataset of rivers with associated gauge measures.

    Returns
    -------
    List[GroundTruthMeasurement]
        List containing three measurements for river lengths 3, 10, and 20.
    """
    return [
        GroundTruthMeasurement(ground_truth=river_with_length(3), gauge_measurement=1),
        GroundTruthMeasurement(ground_truth=river_with_length(10), gauge_measurement=2),
        GroundTruthMeasurement(ground_truth=river_with_length(20), gauge_measurement=3)
    ]