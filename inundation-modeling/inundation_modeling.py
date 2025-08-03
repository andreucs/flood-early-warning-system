from thresholding import ThresholdingModel
from laplace_solver import LaplaceDepthSolverConfig
from manifold import ManifoldModel

from utils import (
    generate_dummy_DEM,
    generate_dummy_train_data,
)

from plotutils import (
    plot_train_data,
    plot_dem,
    plot_thresholding_model,
    plot_manifold_model, 
    plot_single_manifold
)

GROUND_TRUTH = generate_dummy_train_data()
DEM = generate_dummy_DEM()

plot_dem(DEM, figsize=(8,8))
plot_train_data(GROUND_TRUTH, figsize=(19, 14))

tm = ThresholdingModel()
tm.train(GROUND_TRUTH)
inference_maps_tm = [(tm.infer(gauge_level), gauge_level) for gauge_level in [1, 1.5, 2]]
plot_thresholding_model(inference_maps_tm, figsize=(19, 14))

mm = ManifoldModel(
    dem=DEM, 
    scale=100, 
    laplace_config=LaplaceDepthSolverConfig(
        down_scale_factor=8, 
        solve_iterations_factor=3.,
        force_coeff=0.9,
        drop_iterations=1,
        drop_coeff=0.00003),
    force_tolerance=1, 
    force_local_region_width=5,
    flood_agree_threshold=0.1)
mm.train(GROUND_TRUTH)
inference_maps_mm = [(mm.infer(gauge_level), gauge_level) for gauge_level in [1, 1.5, 2]]
plot_manifold_model(inference_maps_mm, figsize=(21, 6))

plot_single_manifold(mm.infer(7.), level =7, figsize=(8, 8))