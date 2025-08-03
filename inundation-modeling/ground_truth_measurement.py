import dataclasses
import numpy as np

# Stores the ground truth image with the corresponding gauge measurement.
@dataclasses.dataclass
class GroundTruthMeasurement:
  # The ground truth inundation map.
  ground_truth: np.ma.MaskedArray
  # The corresponding gauge measurement.
  gauge_measurement: float