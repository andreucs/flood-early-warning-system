from thresholding import ThresholdingModel
from laplace_solver import LaplaceDepthSolverConfig
from ground_truth_measurement import GroundTruthMeasurement
from flood_extend_to_depth import flood_extent_to_depth_solve, flood_height_raster
from typing import Sequence, Optional, Tuple
from absl import logging

import dataclasses
import numpy as np

# Range above heighest measurement and below lowest measurement in which we
# expect to get measurements.
_MEASUREMENT_BUFFER_METERS = 10

# A number that is smaller than the uncertainty bound of a gauge measurement.
_EPSILON = 0.002

def get_cutoff_points(
    measurements: np.ndarray,
    precision_oriented: bool) -> Tuple[Sequence[int], np.ndarray]:
  """Gets all valid cutoff points and threshold values.

  A cutoff point is the index of a measurement at which we can separate lower
  measurements from higher measurements. For a cutoff point i, the matching
  threshold value must be greater or equal to measurement[i-1] and less than
  measurement[i]. Thus, the i'th event is the first wet event.

  The corresponding threshold value is a gauge measurement value which separates
  examples up to the given cutoff from those above the given cutoff.

  Args:
    measurements: All obtained measurements in the training set, sorted in
      ascending order.
    precision_oriented: if true, select the i'th cutoff to be measurements[i] -
      _EPSILON, otherwise the i'th cutoff is measurements[i-1]

  Raises:
    ValueError: If measurements are not sorted.

  Returns:
    cutoff_points: Indices at which there exists a threshold which separates
      between positive and negative measurements. The length of the list is the
      number of distinct measurements plus 1.
    threshold_values: Gauge measurement values which separate at those cutoffs.
      Shape is (num_cutoffs,).
  """
  # `measurements` is sorted in ascending order. Let's just verify this.
  if np.any(np.diff(measurements) < 0):
    raise ValueError('Expected measurements to be sorted: %r' % measurements)

  # Add an artificially low threshold value, to be used for pixels which should
  # always be wet.
  threshold_values = [measurements[0] - _MEASUREMENT_BUFFER_METERS]
  cutoff_points = [0]

  differing_measurements = ~np.isclose(measurements[:-1], measurements[1:])
  differing_locations = np.nonzero(differing_measurements)[0]
  cutoff_points.extend(differing_locations + 1)
  # For cutoffs higher than the lowest event and lower than the highest event,
  # any threshold value between measurements[i] and measurements[i+1] can be
  # used.
  if precision_oriented:
    # To be as conservative as possible, we choose measurements[i+1] - _EPSILON,
    # in order to include as small a region as possible in the risk map.
    threshold_values.extend(measurements[differing_locations + 1] - _EPSILON)
  else:
    # To alert as much as possible, we choose measurements[i],
    # in order to include as large a region as possible in the risk map.
    threshold_values.extend(measurements[differing_locations])

  # Add the final threshold. When not precision oriented, above the highest
  # event all pixels should be considered inundated. When precision oriented,
  # above the inundation map should be equal to that of the highest event.
  threshold_values.append(measurements[-1])
  if precision_oriented:
    threshold_values[-1] += _MEASUREMENT_BUFFER_METERS
  cutoff_points.append(len(measurements))

  logging.info('Found %d threshold values: %r', len(threshold_values),
               np.asarray(threshold_values))

  return cutoff_points, np.asarray(threshold_values, dtype=float)


def _count_true_in_suffixes(imaps: np.ndarray,
                            cutoff_points: Sequence[int]) -> np.ndarray:
  """Returns the number of True's above the cutoff at each pixel.

  Args:
    imaps: Observed inundation maps for all training set flood events, sorted by
      increasing gauge measurement. Shape is (num_train_examples,height,width)
      for 3d arrays, or (num_train_examples,) for 1d arrays.
    cutoff_points: Indices into the first axis of `imaps`, at which it is
      possible to separate between lower and higher flood events. This includes
      all indices except those having identical gauge measurements as the
      subsequent event.

  Returns:
    Array with the same shape as imaps, where the first axis has shape
    num_cutoffs instead of num_train_examples. For example, if imaps is an array
    with shape (num_train_examples, height, width), the result is an array with
    shape (num_cutoffs, height, width) whose [c,i,j] element is the number of
    events whose pixel [i,j] is True above cutoff c.
  """
  all_count_true = np.nancumsum(imaps[::-1], axis=0)[::-1]
  # If the suffix is empty, the number of True's is zero.
  filler_zeros = np.expand_dims(np.zeros_like(all_count_true[0]), 0)
  all_count_true = np.concatenate([all_count_true, filler_zeros])
  return all_count_true.take(cutoff_points, axis=0, mode='clip')


def count_true_wets_per_cutoff(imaps: np.ndarray,
                               cutoff_points: Sequence[int]) -> np.ndarray:
  """Returns the number of wets observations of each pixel above each cutoff.

  Wet observations of a pixel above a given cutoff can be thought of as "true
  wet" decisions for that cutoff value.

  Args:
    imaps: Observed inundation maps for all training set flood events, sorted by
      increasing gauge measurement. Shape is (num_train_examples,height,width)
      for 3d arrays, or (num_train_examples,) for 1d arrays.
    cutoff_points: Indices into the first axis of `imaps`, at which it is
      possible to separate between lower and higher flood events. This includes
      all indices except those having identical gauge measurements as the
      subsequent event.

  Returns:
    Array with the same shape as imaps, where the first axis has shape
    num_cutoffs instead of num_train_examples. For example, if imaps is an array
    with shape (num_train_examples, height, width), the result is an array with
    shape (num_cutoffs, height, width) whose [c,i,j] element is the number of
    events whose pixel [i,j] is wet above cutoff c.
  """
  return _count_true_in_suffixes(imaps, cutoff_points)


def count_false_wets_per_cutoff(imaps: np.ndarray,
                                cutoff_points: Sequence[int]) -> np.ndarray:
  """Returns the number of dry observations of each pixel above each cutoff.

  Dry observations of a pixel above a given cutoff can be thought of as "false
  wet" decisions for that cutoff value.

  Args:
    imaps: Observed inundation maps for all training set flood events, sorted by
      increasing gauge measurement. Shape is (num_train_examples,height,width)
      for 3d arrays, or (num_train_examples,) for 1d arrays.
    cutoff_points: Indices into the first axis of `imaps`, at which it is
      possible to separate between lower and higher flood events. This includes
      all indices except those having identical gauge measurements as the
      subsequent event.

  Returns:
    Array with the same shape as imaps, where the first axis has shape
    num_cutoffs instead of num_train_examples. For example, if imaps is an array
    with shape (num_train_examples, height, width), the result is an array with
    shape (num_cutoffs, height, width) whose [c,i,j] element is the number of
    events whose pixel [i,j] is wet below cutoff c.
  """
  not_imaps = 1 - imaps
  return _count_true_in_suffixes(not_imaps, cutoff_points)


def count_false_drys_per_cutoff(imaps: np.ndarray,
                                cutoff_points: Sequence[int]) -> np.ndarray:
  """Returns the number of wet observations of each pixel below each cutoff.

  Wet observations of a pixel below a given cutoff can be thought of as "false
  dry" decisions for that cutoff value.

  Args:
    imaps: Observed inundation maps for all training set flood events, sorted by
      increasing gauge measurement. Shape is (num_train_examples,height,width)
      for 3d arrays, or (num_train_examples,) for 1d arrays.
    cutoff_points: Indices into the first axis of `imaps`, at which it is
      possible to separate between lower and higher flood events. This includes
      all indices except those having identical gauge measurements as the
      subsequent event.

  Returns:
    Array with the same shape as imaps, where the first axis has shape
    num_cutoffs instead of num_train_examples .For example, if imaps is an array
    with shape (num_train_examples, height, width), the result is an array with
    shape (num_cutoffs, height, width) whose [c,i,j] element is the number of
    events whose pixel [i,j] is dry above cutoff c.
  """
  all_count_false_drys = np.nancumsum(imaps, axis=0)
  # If the cutoff is lower than all events, the number of false drys is zero.
  filler_zeros = np.expand_dims(np.zeros_like(all_count_false_drys[0]), 0)
  all_count_false_drys = np.concatenate([filler_zeros, all_count_false_drys])
  return all_count_false_drys.take(cutoff_points, axis=0, mode='clip')


def _get_pixel_threshold_index(pixel_events: np.ndarray,
                               cutoff_points: np.ndarray,
                               min_ratio: float) -> int:
  """Calculates the threshold index in cutoff_points for a single pixel.

  Args:
    pixel_events: 1D array of the inundation at the pixel for all events, by
      increasing gauge order.
    cutoff_points: Indices into the first axis of `imaps`, at which it is
      possible to separate between lower and higher flood events. This includes
      all indices except those having identical gauge measurements as the
      subsequent event.
    min_ratio: Select the threshold at each pixel such that the ratio between
      added true positives and added false positives is above min_ratio.

  Returns:
    Index within cutoff_points of the threshold of the given pixel.
  """
  while np.nansum(pixel_events):
    true_wets = count_true_wets_per_cutoff(pixel_events, cutoff_points)
    false_wets = count_false_wets_per_cutoff(pixel_events, cutoff_points)
    ratios = true_wets / false_wets
    # The empty slice, corresponding to ratios[-1], has 0 true wets and 0 false
    # wets. We define the ratio there to be 0, as we want any slice that
    # contain a true wet to have a higher ratio than the empty slice.
    ratios[-1] = 0
    # Take the last maximum. In the case we have d, nan, w, we want the
    # threshold to be between nan and w, despite the fact that the threshold
    # between d and nan has the same ratio.
    best_index = ratios.shape[0] - 1 - np.nanargmax(ratios[::-1])

    if ratios[best_index] < min_ratio:
      break
    # Find the next candidate threshold on the prefix of all events below the
    # current best cutoff point.
    best_cutoff_point = cutoff_points[best_index]
    pixel_events = pixel_events[:best_cutoff_point]
    cutoff_points = cutoff_points[:best_index + 1]
  # If the remaining slice contains only NaNs, pixel should always be
  # inundated.
  if np.all(np.isnan(pixel_events)):
    return 0
  return len(cutoff_points) - 1


def _get_threshold_indices(imaps: np.ndarray, cutoff_points: np.ndarray,
                           min_ratio: float) -> np.ndarray:
  """Calculates the threshold index in cutoff_points for all pixels.

  Args:
    imaps: Observed inundation maps for all training set flood events, sorted by
      increasing gauge measurement. Shape is (num_train_examples,height,width).
    cutoff_points: Indices into the first axis of `imaps`, at which it is
      possible to separate between lower and higher flood events. This includes
      all indices except those having identical gauge measurements as the
      subsequent event.
    min_ratio: Select the threshold at each pixel such that the ratio between
      added true positives and added false positives is above min_ratio.

  Returns:
    threshold_indices: Array of threshold indices within cutoff_points for
      each pixel. Shape is (height, width).
  """
  threshold_indices = np.zeros_like(imaps[0, :, :], dtype=int)
  for idx_y in range(threshold_indices.shape[0]):
    for idx_x in range(threshold_indices.shape[1]):
      threshold_indices[idx_y, idx_x] = _get_pixel_threshold_index(
          imaps[:, idx_y, idx_x], cutoff_points, min_ratio)
  return threshold_indices


def _learn_optimal_sar_prediction_internal(imaps: np.ndarray,
                                           measurements: np.ndarray,
                                           min_ratio: float) -> np.ndarray:
  """A version of learn_sar_prediction without the external data structures.

  Selects the threshold such that the number of added true positives divided by
  the number of added false positives is at least min_ratio.

  Args:
    imaps: Observed inundation maps for all training set flood events, sorted by
      increasing gauge measurement. Shape is (num_train_examples,height,width).
    measurements: All obtained measurements in the training set, sorted in
      ascending order.
    min_ratio: Selects the threshold at each pixel such that the ratio between
      added true positives and added false positives is above min_ratio.

  Returns:
    Array of thresholds for each pixel. Shape is (height, width).
  """
  cutoff_points, threshold_values = get_cutoff_points(
      measurements, precision_oriented=True)

  threshold_indices = _get_threshold_indices(imaps, np.array(cutoff_points),
                                             min_ratio)
  thresholds = threshold_values[threshold_indices]

  if np.issubdtype(imaps.dtype, np.floating):
    # If all inundation maps are NaN for a given pixel, make the threshold for
    # that pixel a NaN as well. This occurs when the pixel is outside the
    # requested forecast_region.
    thresholds = np.where(np.all(np.isnan(imaps), axis=0), np.nan, thresholds)

  return thresholds


def masked_array_to_float_array(masked_array):
  float_array = np.array(masked_array, dtype=float)
  mask = np.ma.getmaskarray(masked_array)
  float_array[mask] = np.nan
  return float_array


def learn_optimal_sar_prediction_from_ground_truth(
    flood_events_train: Sequence[GroundTruthMeasurement],
    min_ratio: Optional[float] = None) -> np.ndarray:
  """Constructs a GaugeThresholdingModel based on historical observations.

  This finds the optimal thresholding model for a specific min ratio, as
  described in the paper.

  Args:
    flood_events_train: Sequence of GroundTruthMeasurement objects to be used
      as training set.
    min_ratio: float argument. If provided, selects the threshold at each pixel
      such that the ratio between added true positives and added false positives
      is above min_ratio.

  Returns:
    GaugeThresholdingModel proto containing the learned model. For the optimal
    thresholding model only one threshold is learned, so the high, medium and
    low risk thresholds of the model are identical.
  """

  # shape=(num_train_examples, height, width)
  imaps = np.asarray([
      masked_array_to_float_array(fe.ground_truth) for fe in flood_events_train
  ])

  # shape=(num_train_examples,)
  measurements = np.asarray(
      [fe.gauge_measurement for fe in flood_events_train])
  return _learn_optimal_sar_prediction_internal(imaps, measurements,
                                                                min_ratio)


@dataclasses.dataclass
class _TrainExample:
  """Helper class for holding per-train-example data."""
  height_raster: np.ndarray
  gauge_level: float


class ManifoldModel:
    
  def __init__(self, dem: np.ndarray, scale: float,
               laplace_config: LaplaceDepthSolverConfig, 
               force_tolerance: float, 
               force_local_region_width: int,
               flood_agree_threshold: float) -> None:
    """Initializes a ManifoldModel object.

    Args:
      dem: A 2D array representing the DEM, in the shape of the ground truth
        images.
      scale: The scale of the DEM and the ground truth images.
      laplace_config: The Laplace solver config to use.
      force_tolerance: see `CatScanForceCalculator.tolerance`.
      force_local_region_width: see `CatScanForceCalculator.local_region_width`.
      agree_threshold: Used for the flood-fill algorithm. See 
        `flood_height_raster` for more details.
    """
    self.dem = dem
    self.scale = scale
    self.laplace_config = laplace_config
    self.force_tolerance = force_tolerance
    self.force_local_region_width = force_local_region_width
    self.flood_agree_threshold = flood_agree_threshold
    self.thresholding_model = ThresholdingModel()

  def train(self, ground_truth: Sequence[GroundTruthMeasurement]):
    print("Training an inner thresholding model used for flood-fill.")
    self.thresholding_model.train(ground_truth)
    
    print("Running flood extent to depth on ground truth examples..")
    sorted_ground_truth = sorted(ground_truth, 
                                 key=lambda gt: gt.gauge_measurement)
    self._train_examples = []
    for gt in sorted_ground_truth:
      gauge_level = gt.gauge_measurement
      print('Running flood extent to depth algorithm for image at gauge_level',
            gauge_level)
      height_raster = flood_extent_to_depth_solve(gt.ground_truth,
                                                  self.dem,
                                                  self.scale,
                                                  self.laplace_config,
                                                  self.force_tolerance,
                                                  self.force_local_region_width)
      self._train_examples.append(_TrainExample(height_raster=height_raster,
                                                gauge_level=gauge_level))
      
  def _interpolate_between(self, level: float, example_below: _TrainExample,
                           example_above: _TrainExample) -> np.ndarray:
    """Linearly interpolates between two train examples.

    Performs the piecewise linear interpolation on the train examples.

    Args:
      level: The gauge level to be used.
      example_below: The train example which is closest to `level` from below.
      example_above: The train example which is closest to `level` from above.

    Returns:
      The water height raster which is the linear interpolation between the
      provided train examples.
    """
    level_below = example_below.gauge_level
    level_above = example_above.gauge_level
    level_ratio = (level - level_below) / (level_above - level_below)
    return (level_ratio * example_above.height_raster +
            (1 - level_ratio) * example_below.height_raster)

  def _infer_piecewise_linear_manifold(self, gauge_level: float) -> np.ndarray:
    """Infers piecewise linear water height manifold given gauge level.

    The method performs piecewise linear interpolation between the saved train
    examples.

    Args:
      gauge_level: The gauge level to infer for.

    Returns:
      The inferred low-resolution water height manifold.
    """
    train_levels = [example.gauge_level for example in self._train_examples]
    index = np.searchsorted(train_levels, gauge_level)

    if index == 0:
      # Lower than train event.
      lowest_example = self._train_examples[0]
      return lowest_example.height_raster

    elif index == len(train_levels):
      # Extreme event.
      highest_example = self._train_examples[-1]
      # Create a new height raster by adding the difference between the current
      # measurement and the (previous) highest measurement to the height raster
      # of the (previous) highest measurement.
      return highest_example.height_raster + (gauge_level - train_levels[-1])
    else:
      return self._interpolate_between(
          level=gauge_level,
          example_below=self._train_examples[index - 1],
          example_above=self._train_examples[index])

  def infer(self, gauge_level: float):
    reference_inundation_map = self.thresholding_model.infer(gauge_level)
    manifold = self._infer_piecewise_linear_manifold(gauge_level)
    return flood_height_raster(interpolated_height_raster=manifold, 
                               dem=self.dem,
                               inundation_map=reference_inundation_map,
                               agree_threshold=self.flood_agree_threshold)