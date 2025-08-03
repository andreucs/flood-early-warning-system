from typing import Tuple
from absl import logging
from laplace_solver import LaplaceDepthSolverConfig
from laplace_solver import CatScanForceCalculator
from laplace_solver import laplace_depth_solve

import numpy as np
import PIL.Image
import scipy.ndimage
import skimage.morphology

# The minimimum size of a flooded components to be regarded by the
# flood_extent_to_depth algorithm. Smaller flooded components will be removed
# from the input inundation map.
_MIN_OBJECT_AREA_KM2 = 0.1


def down_scale(raster: np.ndarray, down_scale_factor: int, 
               fraction: float = 0.1) -> np.ndarray:
  """Down scale a boolean raster image by down_scale_factor."""
  b = raster.shape[0] // down_scale_factor
  return raster.reshape(-1, down_scale_factor, b, down_scale_factor).mean(
      (-1, -3)) > fraction


def clean_inundation_map(inundation_map: np.ma.MaskedArray,
                         scale_meters: float) -> np.ma.MaskedArray:
  """Removes small objects and holes from inundation map.

  Args:
    inundation_map: The inundation map to process, as a boolean 2D masked array.
    scale_meters: The inundation map scale, i.e. meters per pixel.

  Returns:
    A masked array representing the input inundation map without the small
    objects and holes.
  """
  min_object_area_pixels = 1000000 * _MIN_OBJECT_AREA_KM2 / scale_meters**2
  filtered_inundation_map = skimage.morphology.remove_small_objects(
      np.ma.filled(inundation_map.astype(bool), False),
      min_size=min_object_area_pixels)
  skimage.morphology.remove_small_holes(
      filtered_inundation_map,
      area_threshold=min_object_area_pixels)
  return np.ma.masked_where(np.isnan(inundation_map),
                            filtered_inundation_map)


def _nearest_neighbhor_interpolation(
    partial_image: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray]:
  """Interpolates a partial image to a full one using nearest neighbors.

  This function gets a partial 2D image, i.e. an image with missing pixels, and
  interpolates them according to the nearest neighbour algorithm, i.e. the value
  at a missing pixel is taken from its nearst valid pixel.

  Args:
    partial_image: A masked array, where missing pixels are masked out.

  Returns:
    full_image: The output interpolated image, that has no invalid pixels.
    interpolation_distance_pixels: A 2D numpy array with the same shape of the
      input image that holds for every pixel the distance (in pixels) to a
      known pixel.
  """
  edt_distances, edt_indices = scipy.ndimage.morphology.distance_transform_edt(
      input=np.ma.getmaskarray(partial_image),
      return_indices=True,
      return_distances=True)
  full_image = np.ma.getdata(partial_image)[edt_indices[0], edt_indices[1]]

  return full_image, edt_distances


def extend_height_raster(
    partial_height_raster: np.ma.MaskedArray, down_scale_factor: int,
    input_inundation_map: np.ma.MaskedArray,
    inundation_map_scale: int) -> Tuple[np.ndarray, np.ndarray]:
  """Extends a height raster to match valid pixels in the inundation map.

  Args:
    partial_height_raster: A masked array, where missing pixels are masked out.
    down_scale_factor: The ratio between the partial_height_raster scale to the
      input_inundation_map scale.
    input_inundation_map: The inundation map used to generate the
      partial_height_raster. It is used to determine which of the pixels of the
      output height_raster are valid.
    inundation_map_scale: The scale in meters of the input_inundation_map.

  Returns:
    height_raster: The output interpolated height raster, valid at the same
      pixels the input_inundation_map is valid.
    interpolation_distance: A 2D numpy array in the same shape of the
      height_raster that holds for every pixel the distance (in meters) to a
      known pixel.
  """

  # interpolate the height raster to the whole image.
  height_raster, interpolation_distance_pixels = (
      _nearest_neighbhor_interpolation(partial_height_raster))

  # Convert the interpolation_distance to be in meters rather than pixels.
  height_raster_scale = inundation_map_scale * down_scale_factor
  interpolation_distance = interpolation_distance_pixels * height_raster_scale

  # Invalidate pixels that are in an invalid part of the inundation map. Make
  # sure that no new invalid pixels are introduced due to the down scale by
  # invalidating a low-res pixel only if *all* the high-res pixels are invalid.
  fraction = 1 - 1 / down_scale_factor ** 2
  invalid_part = down_scale(
      np.ma.getmaskarray(input_inundation_map),
      down_scale_factor=down_scale_factor, fraction=fraction)
  height_raster[invalid_part] = np.nan

  return height_raster, interpolation_distance


def _keep_flooded_components(water_above_dem: np.ndarray,
                             known_inundation_map: np.ndarray,
                             agree_threshold: float) -> np.ndarray:
  """Given a possible inundation map, returns only the flooded component.

  Chooses only flooded compoenents from `water_above_dem` that agree with
  `known_inundation_map`.

  Args:
    water_above_dem: A boolean map that indicates for every pixel whether its
      calculated water height is above the DEM or below.
    known_inundation_map: An inundation map to be used as reference. The flooded
      components are chosen to be components that appear in this inundation map.
    agree_threshold: A number between 0 and 1 that indicates what is the
      minimum agreement ratio for a component to be chosen to be flooded.

  Returns:
    The output image containing only water_above_dem components that agree with
    known_inundation_map at at least `agree_threshold` of the pixels.
  """
  # water_above_dem_components indicates, for each pixel in water_above_dem,
  # the index of the connected component to which it belongs.
  water_above_dem_components = skimage.measure.label(
      water_above_dem, background=0, connectivity=1)
  num_components = np.max(water_above_dem_components) + 1

  if num_components == 0:
    return water_above_dem

  component_size, bins = np.histogram(
      water_above_dem_components,
      bins=num_components-1,
      range=(1, num_components))

  # For each connected component, find its agree ratio with
  # known_inundation_map.
  agree_image = np.where(known_inundation_map, water_above_dem_components, 0)
  components_agree, _ = np.histogram(
      agree_image, bins=num_components-1, range=(1, num_components))
  component_agree_ratio = dict(
      zip(bins[:-1], components_agree / component_size))

  # Keep only components that have more than agree_threshold agreement with the
  # known_inundation_map.
  good_components = [
      round(c)
      for c, ratio in component_agree_ratio.items()
      if ratio > agree_threshold
  ]
  return np.isin(water_above_dem_components, good_components)


def flood_height_raster(interpolated_height_raster: np.ndarray,
                        inundation_map: np.ma.MaskedArray, dem: np.ndarray,
                        agree_threshold: float) -> np.ndarray:
  """Given water height raster and a DEM, calculate the boolean inundation map.

  Args:
    interpolated_height_raster: 2D numpy array with water level in meters above
      see level. This array can have any scale, and it will be scaled to the DEM
      shape.
    inundation_map: An inundation map to be used for finding the start points
      for the flood fill algorithm. For more information, refer to
      `_keep_flooded_components`. This array can have any scale and it will be
      scaled to the DEM shape.
    dem: The DEM to be used for flood, as a 2D numpy array.
    agree_threshold: A number between 0 and 1 that indicates what is the
      minimum agreement ratio between the output and the input inundation maps
      for a component to be chosen to be flooded. For more information, refer to
      `_keep_flooded_components`.

  Returns:
    The flooded inundation map, in the DEM scale.
  """

  # Change inputs to DEM scale.
  height_raster_upscale = PIL.Image.fromarray(
      interpolated_height_raster).resize(dem.T.shape, 
                                         resample=PIL.Image.BILINEAR)
  height_raster_mask_upscale = PIL.Image.fromarray(
      np.ma.getmaskarray(inundation_map)).resize(dem.T.shape)
  inundation_map_upscale = PIL.Image.fromarray(inundation_map).resize(
      dem.T.shape)

  # Calculate boolean inundation map.
  water_above_dem = (height_raster_upscale > dem)
  water_above_dem[height_raster_mask_upscale] = False
  flood_image = _keep_flooded_components(
      water_above_dem, inundation_map_upscale, agree_threshold=agree_threshold)

  depth_image = (height_raster_upscale - dem)
  depth_image[~flood_image] = 0

  overall_mask = np.logical_or(height_raster_mask_upscale, np.isnan(dem))
  return np.ma.MaskedArray(depth_image, mask=overall_mask)


def flood_extent_to_depth_solve(inundation_map: np.ma.MaskedArray, 
                                dem: np.ndarray, scale: int,
                                laplace_config: LaplaceDepthSolverConfig,
                                force_tolerance: float,
                                force_local_region_width: int) -> np.ndarray:
  """Extracts water height from flood extent image and DEM.

  Args:
    inundation_map: The inundation map as a boolean 2D array from which to
      extract the water height. Missing values are expected to be masked out.
    dem: The DEM to be used, as a 2D float numpy array. It's proportions must
      match the inundation map. Invalid values are expected to be np.nan.
    scale: The maps scale, in meters for pixel.
    config: The solve configuration. Please refer to `LaplaceDepthSolverConfig`
      for more details.
    force_tolerance: see `CatScanForceCalculator.tolerance`.
    force_local_region_width: see `CatScanForceCalculator.local_region_width`.

  Returns:
  The result water height.
  """

  logging.info('Starting initial processing of the inundation map.')
  processed_inundation_map = clean_inundation_map(inundation_map, scale)
  down_scale_factor = laplace_config.down_scale_factor

  force_calculator = CatScanForceCalculator(
        dem=dem,
        tolerance=force_tolerance,
        local_region_width=force_local_region_width)
  
  logging.info('Running laplace solver on wet pixels...')
  partial_height_raster = laplace_depth_solve(
      inundation_map=processed_inundation_map,
      force_calculator=force_calculator,
      config=laplace_config)
  
  logging.info('Extending height raster to the entire image...')
  height_raster, _ = extend_height_raster(
      partial_height_raster=partial_height_raster,
      down_scale_factor=down_scale_factor,
      input_inundation_map=inundation_map,
      inundation_map_scale=scale)

  return height_raster