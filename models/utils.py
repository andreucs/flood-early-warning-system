import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
import polars as pl
import datetime
import config

def get_basins(
        txt_path: Union[str, Path] = config.BASINS_DIR) -> list[str]:
    """
    Load basin identifiers from a text file, one per line.

    Parameters
    ----------
    txt_file : Union[str, Path]
        Path to the .txt file containing basin identifiers, one per line.

    Returns
    -------
    list[str]
        List of basin identifiers.
    """
    
    txt_path = Path(txt_path)
    if not txt_path.exists() or not txt_path.is_file():
        raise ValueError(f"The file {txt_path!r} does not exist or is not a file.")
    
    with txt_path.open(mode="r", encoding="utf-8") as f:
        basins = [line.strip() for line in f if line.strip()]

    return basins


def load_static_attributes(
        attributes_dir: Union[str, Path] = config.ATTRIBUTES_DIR,
        static_vars: List[str] = config.STATIC_INPUTS) -> pl.DataFrame:
    """
    Load static attributes for specified basins from multiple CSV files.This function
    scans the given directory for CSV files containing basin attributes, filters each
    file to include only the basins listed in `basins`, and extracts the requested columns.

    Parameters
    ----------
    attributes_dir : Union[str, Path]
        Path to the directory containing CSV files with basin attributes.
    variables : List[str]
        List of attribute column names to extract from each CSV.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame where:
          - `unique_id` holds each basin identifier.
          - Each column corresponds to one of the requested variables.
    """

    basins = get_basins()
    attr_path = Path(attributes_dir)

    if not attr_path.exists() or not attr_path.is_dir():
            raise ValueError(f"The directory {attributes_dir!r} does not exist or is not a directory.")

    static_df = pl.DataFrame({"gauge_id": list(basins)})

    for csv_file in attr_path.glob("*.csv"):
        df_attr = pl.read_csv(str(csv_file), infer_schema_length=10_000)

        if "gauge_id" not in df_attr.columns:
            continue

        df_filtered_by_basin = df_attr.filter(pl.col("gauge_id").is_in(basins))

        if df_filtered_by_basin.is_empty():
            continue

        # Determine which of the requested variables appear in this file
        common_vars = [v for v in static_vars if v in df_filtered_by_basin.columns]
        if not common_vars:
            continue

        df_with_requested_static_vars = df_filtered_by_basin.select(["gauge_id", *common_vars])
        static_df = static_df.join(df_with_requested_static_vars, on="gauge_id", how="left")

    static_df = static_df.rename({"gauge_id": "unique_id"})

    return static_df


def load_dynamic_timeseries(
    timeseries_dir: Union[str, Path] = config.TIMESERIES_DIR,
    variables: List[str] = config.HISTORICAL_INPUTS,
    target: str = config.TARGET,
    start_date_filter: str = config.START_DATE) -> pl.DataFrame:
    """
    Load and concatenate dynamic time series for specified basins from multiple CSV files.

    This function reads basin identifiers from a text file, then for each basin it
    loads the corresponding CSV time series file (named
    CAMELS_DE_hydromet_timeseries_{gauge_id}.csv), filters records to those
    on or after `start_date_filter`, checks that the requested dynamic variables and
    target column are present.

    Parameters
    ----------
    timeseries_dir : Union[str, Path]
        Directory path containing the time series CSV files.
    variables : Sequence[str]
        List of dynamic variable column names to extract from each CSV.
    target : str
        Name of the target column to include and rename to 'y'.
    start_date_filter : Union[str, datetime.datetime, None], optional
        Date string (YYYY-MM-DD) or datetime from which to filter records.
        If None, no date filtering is applied.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:
          - `unique_id`: basin identifier
          - `ds`: timestamp column
          - one column per entry in `variables`
          - `y`: target column values
    """
    gauge_ids = get_basins()
    ts_path = Path(timeseries_dir)
    start_date = datetime.datetime.strptime(start_date_filter, "%Y-%m-%d")

    frames: list[pl.DataFrame] = []

    for gid in gauge_ids:
        csv_path = ts_path / f"CAMELS_DE_hydromet_timeseries_{gid}.csv"
        
        df_ts = pl.read_csv(
            csv_path,
            schema_overrides={
                    "date": pl.Date,
                    "discharge_spec_obs": pl.Float64
            },
            infer_schema=10_000
        )
        
        if "date" in df_ts.columns:
            df_ts = df_ts.filter(pl.col("date") >= start_date)
            if df_ts.is_empty():
                continue

        missing = [v for v in variables if v not in df_ts.columns]
        if missing:
            raise ValueError(f"In {csv_path.name}, missing dynamic variables: {missing}")
        if target not in df_ts.columns:
            raise ValueError(f"In {csv_path.name}, missing target column: {target}")
        
        select_cols = []
        if "date" in df_ts.columns:
            select_cols.append("date")
        select_cols.extend(variables)
        select_cols.append(target)
        df_sel = df_ts.select(select_cols)
        
        df_sel = df_sel.with_columns(pl.lit(gid).alias("unique_id"))
        frames.append(df_sel)

    result = pl.concat(frames, how="vertical")
    result = result.rename({
        "date": "ds",
        target: "y"
    })

    return result


def set_trial(
    historical_df: pl.DataFrame,
    static_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Apply trial-mode filtering to both historical and static DataFrames.

    If `config.TRIAL` is True, only the first `config.N_BASINS` from `basins` will be
    included. Otherwise, both DataFrames are returned unchanged.

    Parameters
    ----------
    historical_df : pl.DataFrame
        Polars DataFrame with columns ['unique_id', 'ds', ...] where 'ds' is datetime.
    static_df : pl.DataFrame
        Polars DataFrame with column 'unique_id' and static attributes.
    basins : Sequence[str]
        Ordered list of basin identifiers.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        `(filtered_historical, filtered_static)`. In trial mode, both are
        subset to the trial basins.
    """

    basins = get_basins()
    trial_basins = basins[: config.N_BASINS]

    hist_filtered = historical_df.filter(
        pl.col("unique_id").is_in(trial_basins)
    )
    stat_filtered = static_df.filter(
        pl.col("unique_id").is_in(trial_basins)
    )

    return hist_filtered, stat_filtered


def save_results(
    fcst_df: pl.DataFrame,
    filename: str,
    model_name: str,
    loss: str = config.LOSS,
    results_dir: Union[str, Path] = None) -> Path:
    """
    Rename forecast DataFrame columns, ensure the results directory exists,
    and save the DataFrame to a Parquet file.

    Parameters
    ----------
    fcst_df : pl.DataFrame
        Polars DataFrame with forecast columns including:
        'unique_id', 'y', '{model_name}-median', '{model_name}-lo-90', '{model_name}-hi-90'.
    results_dir : Union[str, Path], optional
        Directory in which to save results. If None, defaults to
        '<project_root>/results' where project_root is two levels above this file.
    filename : str, default "{model_name}.parquet"
        Name of the Parquet file to write.
    """

    if loss == "mqloss":
        renamed = fcst_df.rename({
            "unique_id": "basins",
            "y": "obs",
            f"{model_name}-median": "sim",
            f"{model_name}-lo-90": "sim-lo-90",
            f"{model_name}-hi-90": "sim-hi-90"
        })
    else:
        renamed = fcst_df.rename({
            "unique_id": "basins",
            "y": "obs",
            f"{model_name}": "sim",
        })

    if results_dir is None:
        # Default: two levels up from this file, then "results"
        project_root = Path(__file__).resolve().parent.parent
        results_dir = project_root / "results"
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    output_file = results_path / filename
    renamed.write_parquet(output_file, statistics=False)