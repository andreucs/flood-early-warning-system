from typing import Tuple, Dict
from matplotlib.font_manager import fontManager
from matplotlib import rcParams
from matplotlib.ticker import NullLocator
from matplotlib.dates import DateFormatter, MonthLocator
from pathlib import Path
from pyfonts import load_google_font
from evalutils import split_by_timestep

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import datetime
import pandas as pd
import locale

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
        "legend.fontsize": 18,
        "figure.titlesize": 20
    })


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate empirical cummulative density function
    
    Parameters
    ----------
    x : np.ndarray
        Array containing the data
    
    Returns
    -------
    x : np.ndarray
        Array containing the sorted metric values
    y : np.ndarray
        Array containing the sorted cdf values
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def plot_cdf(
    metric: str,
    day: int,
    figsize: Tuple[int, int],
    mse_loss: Dict[str, Dict[str, pl.DataFrame]],
    mqloss: Dict[str, Dict[str, pl.DataFrame]]
) -> None:
    """
    Plots the Cumulative Distribution Function (CDF) of MSE and MQ metrics for different models on a specific day.
    
    Args:
        metric (str): The name of the performance metric to be displayed on the x-axis.
        day (int): The specific day for which the data should be plotted.
        figsize (Tuple[int, int]): The size of the figure (width, height).
        mse_loss (dict): Dictionary of MSE loss data per day and model using polars DataFrames.
        mqloss (dict): Dictionary of MQ loss data per day and model using polars DataFrames.
    
    Returns:
        None: The function saves the plot as a PDF file.
    """
    
    model_draw_style = {
        'lstm': {
            'mse_color': '#1b9e77',
            'mq_color': '#b3e2cd',
            'mse_linestyle': '-',
            'mq_linestyle': '--',
            'mse_label': 'LSTM MSE',
            'mq_label': 'LSTM MQ'
        },
        'tcn': {
            'mse_color': '#7570b3',
            'mq_color': '#cbd5e8',
            'mse_linestyle': '-',
            'mq_linestyle': '--',
            'mse_label': 'TCN MSE',
            'mq_label': 'TCN MQ'
        },
        'patchtst': {
            'mse_color': '#d95f02',
            'mq_color': '#fdcdac',
            'mse_linestyle': '-',
            'mq_linestyle': '--',
            'mse_label': 'PatchTST MSE',
            'mq_label': 'PatchTST MQ'
        }
    }

    mse_metrics_filtered_by_day = mse_loss[f"day{day}"]
    mq_metrics_filtered_by_day = mqloss[f"day{day}"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.2)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.10)
    ax.set_axisbelow(True)

    for model_type, df in mse_metrics_filtered_by_day.items():
        bin_, cdf_ = ecdf(df["NSE"].to_numpy())
        ax.plot(
            bin_,
            cdf_,
            label=model_draw_style[model_type]['mse_label'],
            color=model_draw_style[model_type]['mse_color'],
            marker="s",
            markevery=5,
            linestyle=model_draw_style[model_type]['mse_linestyle']
        )

    for model_type, df in mq_metrics_filtered_by_day.items():
        bin_, cdf_ = ecdf(df["NSE"].to_numpy())
        ax.plot(
            bin_,
            cdf_,
            label=model_draw_style[model_type]['mq_label'],
            color=model_draw_style[model_type]['mq_color'],
            marker="s",
            markevery=5,
            linestyle=model_draw_style[model_type]['mq_linestyle']
        )

    handles, labels = ax.get_legend_handles_labels()
    desired_order = [1, 4, 2, 5, 0, 3]

    ax.legend(
        [handles[i] for i in desired_order],
        [labels[i] for i in desired_order],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.11),
        ncol=3,
        frameon=False
    )

    ax.set_xlabel(metric)
    ax.set_ylabel('CDF')

    fig.savefig(f"figures/{metric}_day{day}.pdf", bbox_inches='tight', dpi=300, format='pdf')


def plot_simulation(model: str,
                    figsize: Tuple[int, int]) -> None:
    """
    Plots simulated and observed streamflow for a specific basin along with precipitation data and confidence intervals.
    
    Args:
        model (str): Name of the model to load prediction results from.

    Returns:
        None: Saves a PDF plot for the basin.
    """
    # Setup
    BASIN = "DE210310"
    timeseries_dir = "../data/CAMELS_DE/timeseries"
    ts_path = Path(timeseries_dir)
    csv_path = ts_path / f"CAMELS_DE_hydromet_timeseries_{BASIN}.csv"

    start_date = datetime.datetime.strptime("2017-01-01", "%Y-%m-%d")
    end_date = datetime.datetime.strptime("2017-09-30", "%Y-%m-%d")

    # Load timeseries
    df_ts = pl.read_csv(
        csv_path,
        schema_overrides={
            "date": pl.Date,
            "discharge_spec_obs": pl.Float64
        },
        infer_schema=10_000
    )

    if "date" in df_ts.columns:
        df_ts = df_ts.with_columns(pl.col("date").cast(pl.Date, strict=False))
        df_ts = df_ts.filter(
            pl.col("date").is_between(start_date, end_date, closed="both")
        )

    df_ts = df_ts.select(
        pl.col("date"),
        pl.col("precipitation_mean").alias("prep")
    )

    # Load simulation data
    d = split_by_timestep(Path(f"results/mq/{model}.parquet"))
    day1_basin = (
        d["day1"]
        .filter(
            (pl.col("basins") == BASIN)
            & (pl.col("ds").is_between(start_date, end_date))
        )
        .drop("cutoff", "basins")
        .rename({"ds": "date"})
        .join(df_ts, on="date", how="left")
    )

    # Convert to pandas
    pdf = day1_basin.to_pandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.set_index('date').sort_index()

    # Plotting
    fig, ax1 = plt.subplots(figsize=figsize)

    # Grid and styling
    ax1.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.2)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.1)
    ax1.set_axisbelow(True)

    # Main lines and confidence interval
    ax1.plot(
        pdf.index, pdf['sim'],
        label='sim',
        zorder=2,
        color="#ff7f0e",
        linewidth=2,
        linestyle='-'
    )
    ax1.plot(
        pdf.index, pdf['obs'],
        label='obs',
        zorder=1,
        color="#9467bd",
        linewidth=2
    )
    ax1.fill_between(
        pdf.index,
        pdf['sim-lo-90'],
        pdf['sim-hi-90'],
        alpha=0.3,
        label='C.I 90%',
        color="#bdbdbd",
        zorder=3
    )

    ax1.set_ylabel('Streamflow')
    ax1.set_ylim(0, 70)

    # Date formatting
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    ax1.xaxis.set_major_locator(MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax1.xaxis.set_minor_locator(NullLocator())

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    desired_order = [1, 0, 2]
    ax1.legend(
        [handles[i] for i in desired_order],
        [labels[i] for i in desired_order],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False
    )

    # Precipitation (secondary axis)
    ax2 = ax1.twinx()
    ax2.bar(
        pdf.index, pdf['prep'],
        width=1,
        alpha=0.8,
        color="#1f78b4"
    )
    ax2.set_ylabel('Precipitation')
    ax2.set_ylim(0, 180)
    ax2.invert_yaxis()

    # Title and save
    plt.title(f'Basin {BASIN}', loc="left")
    fig.savefig(f"figures/{BASIN}.pdf", bbox_inches='tight', dpi=300, format='pdf')