from pyfonts import load_google_font
from matplotlib.font_manager import fontManager
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from typing import List, Tuple, Union
from ground_truth_measurement import GroundTruthMeasurement

import numpy.ma as ma

import matplotlib.pyplot as plt
import numpy as np

font = load_google_font("Courier Prime", weight="regular", italic=False)
fontManager.addfont(str(font.get_file()))
rcParams.update(
    {
        "font.family": font.get_name(),
        "font.style": font.get_style(),
        "font.weight": font.get_weight(),
        "font.size": font.get_size(),
        "font.stretch": font.get_stretch(),
        "font.variant": font.get_variant(),
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 20
    })


COLORS: List[str] = ["#EEEEEE", "#1F1F1F"]

def gray_scale_cmap(n: int = 256) -> LinearSegmentedColormap:
    """
    Create a continuous colormap shifting from light gray to black.

    Parameters
    ----------
    n : int, optional
        Number of discrete color levels. Defaults to 256.

    Returns
    -------
    LinearSegmentedColormap
        A colormap that interpolates between COLORS list.
    """
    return LinearSegmentedColormap.from_list("GrayScale", COLORS, N=n)



def plot_thresholding_model(
    bitmaps: List[Tuple[np.ndarray, Union[int, float, str]]],
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Save a row of three boolean images from gauge levels.

    Parameters
    ----------
    bitmaps : list of tuples
        Each tuple should contain:
          - a 2D numpy.ndarray of booleans
          - an int, float, or str gauge level for the title
        The list must have exactly three entries.
    figsize : tuple of two ints, optional
        Figure size as (width, height). Defaults to (12, 4).
    """
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    legend_handles = [
        Patch(color=COLORS[0], label='Dry'),
        Patch(color=COLORS[1], label='Wet')
    ]

    for ax, (mat, level) in zip(axes, bitmaps):
        ax.imshow(mat, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Gauge level: {level}', loc='left')
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend(
        handles=legend_handles,
        loc='lower right',
        frameon=True,
        framealpha=1,
        borderaxespad=0.5
    )

    fig.subplots_adjust(wspace=0.05, right=0.98, bottom=0.15)
    
    fig.savefig("figures/thresholding_model_inference_samples.pdf")


def plot_manifold_model(
    manifolds: List[Tuple[ma.MaskedArray, Union[int, float, str]]],
    cmap: Union[str, LinearSegmentedColormap] = gray_scale_cmap(),
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot three depth maps from gauge levels.

    Parameters
    ----------
    manifolds : list of tuples
        Each tuple should contain:
          - a numpy.ma.MaskedArray of depth values (with optional mask)
          - an int, float, or str gauge level for the title
        The list must have exactly three entries.
    cmap : str or Colormap, optional
        Colormap to use for rendering depth values.
        Defaults to the gray-to-blue continuous colormap.
    figsize : tuple of two ints, optional
        Figure size as (width, height). Defaults to (12, 4).
    """
    # Compute global color scale limits
    vmin = min([m.min() for m, _ in manifolds])
    vmax = max([m.max() for m, _ in manifolds])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (manifold_ma, level) in zip(axes, manifolds):
        im = ax.imshow(
            manifold_ma,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            aspect='auto'
        )
        ax.set_title(f'Gauge level: {level}', loc='left')
        ax.set_xticks([])
        ax.set_yticks([])

    # Add shared vertical colorbar
    fig.colorbar(
        im,
        ax=axes,
        orientation='vertical',
        fraction=0.02,
        pad=0.04
    )

    fig.subplots_adjust(wspace=0.05, right=0.88)
    
    fig.savefig("figures/manifold_model_inference_samples.pdf")


def plot_dem(
    dem: np.ndarray,
    cmap: Union[str, LinearSegmentedColormap] = gray_scale_cmap(),
    figsize: Tuple[int, int] = (6, 6)
) -> None:
    """
    Plot a single DEM with a unified color scale and a vertical colorbar.

    Parameters
    ----------
    dem : numpy.ndarray
        2D array representing digital elevation values.
    cmap : str or Colormap, optional
        Colormap to use for rendering elevation. Defaults to the
        gray-to-blue continuous colormap.
    figsize : tuple of two ints, optional
        Figure size as (width, height). Defaults to (6, 6).
    """
    vmin, vmax = dem.min(), dem.max()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        dem,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        aspect='auto'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Digital Elevation Model', loc='left')

    fig.colorbar(
        im,
        ax=ax,
        orientation='vertical',
        fraction=0.046,
        pad=0.04
    )

    fig.savefig("figures/dummy_DEM.pdf")

def plot_train_data(
    data: List[GroundTruthMeasurement],
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot three training samples showing ground truth maps with gauge labels.

    Parameters
    ----------
    data : list of GroundTruthMeasurement
        A list of exactly three training data objects.
    figsize : tuple of two ints, optional
        Figure size as (width, height). Defaults to (12, 4).
    """

    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    legend_handles = [Patch(color=COLORS[0], label='Dry'), Patch(color=COLORS[1], label='Wet')]
    for ax, sample in zip(axes, data):
        ax.imshow(sample.ground_truth, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Gauge level: {sample.gauge_measurement}', loc='left')
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(handles=legend_handles, 
                    loc='lower right',
                    frameon=True, 
                    framealpha=1, 
                    borderaxespad=0.5)
    fig.subplots_adjust(wspace=0.05, right=0.98, bottom=0.15)
    
    fig.savefig("figures/train_data_inundation_modeling.pdf")


def plot_single_manifold(
    manifold: ma.MaskedArray,
    level: Union[int, float, str],
    cmap: Union[str, LinearSegmentedColormap] = gray_scale_cmap(),
    figsize: Tuple[int, int] = (6, 6)
) -> None:
    """
    Plot a single depth map with its gauge level and a vertical colorbar.

    Parameters
    ----------
    manifold : numpy.ma.MaskedArray
        Depth data with an optional mask.
    level : int, float, or str
        Gauge level to display in the title.
    cmap : str or Colormap, optional
        Colormap to represent depth. Defaults to the gray-to-blue map.
    figsize : tuple of two ints, optional
        Figure size as (width, height). Defaults to (6, 6).
    """
    vmin = manifold.min()
    vmax = manifold.max()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        manifold,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        aspect='auto'
    )
    ax.set_title(f'Gauge level: {level}', loc='left')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(
        im,
        ax=ax,
        orientation='vertical',
        fraction=0.046,
        pad=0.04
    )
    
    fig.savefig("figures/level7_manifold.pdf")