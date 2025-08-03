from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
from collections import defaultdict
from metrics import compute_metrics
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import polars as pl
import numpy as np
import pandas as pd
import pathlib

def split_by_timestep(file_path: Union[str, Path]) -> Dict[str, pl.DataFrame]:
    """
    Read model forecast results from a Parquet file, compute
    a prediction horizon for each row (h = ds - cutoff), and split the DataFrame
    into subsets corresponding to each horizon day.

    Parameters
    ----------
    parquet_path : Union[str, Path]
        Path (as a string or pathlib.Path) to the Parquet file containing at least:
        - ds: the date of the forecast target
        - cutoff: the date at which the forecast was made

    Returns
    -------
    Dict[str, pl.DataFrame]
        A dictionary where each key is 'dayN' (N = 1 to max_horizon) and each
        value is the subset of the original DataFrame for that forecast step.
    """
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    df = pl.read_parquet(file_path)

    # Compute horizon h = ds - cutoff
    df = df.with_columns(
        (pl.col("ds").cast(pl.Int16) - pl.col("cutoff").cast(pl.Int16))
        .alias("h")
    )

    # Determine maximum horizon
    max_horizon = df["h"].max()

    # Build a dict of DataFrames per horizon day
    results: Dict[str, pl.DataFrame] = {}
    for timestep in range(1, max_horizon + 1):
        key = f"day{timestep}"
        results[key] = (
            df
            .filter(pl.col("h") == timestep)
            .drop("h")
        )
    return results


def compute_metrics_by_horizon(
    data_dir: Union[str, Path],
    metrics: List[str],
    horizon_days: int
) -> Dict[str, Dict[str, pl.DataFrame]]:
    """
    Process model output files to compute performance metrics for individual models and their ensemble
    across multiple forecast horizons.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to the directory containing Parquet result files for each model.
    metrics : List[str]
        List of metric names to compute (e.g., ["NSE", "KGE", ...]).
    horizon_days : int
        Number of forecast horizons to process (e.g., 3 for day1, day2, day3).

    Returns
    -------
    Dict[str, Dict[str, pl.DataFrame]]
        A dictionary mapping each horizon key (e.g., 'day1') to another dictionary:
        - Keys are model names (file stem) plus an 'ensemble' entry.
        - Values are Polars DataFrames containing metric values per basin.
    """
    results: Dict[str, Dict[str, pl.DataFrame]] = {}
    base_path = Path(data_dir)
    parquet_files = list(base_path.glob("*.parquet"))

    for day_index in range(1, horizon_days + 1):
        day_key = f"day{day_index}"
        results[day_key] = {}
        ensemble_data: Dict[str, Dict[str, pl.DataFrame]] = defaultdict(dict)

        for file_path in parquet_files:
            model_name = Path(file_path).stem
            per_model_rows: List[Dict] = []
            by_timestep = split_by_timestep(file_path)
            df_for_day = by_timestep[day_key]

            for (basin_tuple, df_basin) in df_for_day.group_by("basins"):
                basin = basin_tuple[0]
                obs_series = df_basin["obs"].to_numpy()
                sim_series = df_basin["sim"].to_numpy()
                date_series = df_basin["ds"].to_numpy()
                

                # Accumulate ensemble data
                # if basin in ensemble_data:
                #     prev_df = ensemble_data[basin]["df"]
                #     ensemble_data[basin]["df"] = prev_df.with_columns(
                #         (pl.col("sim") + df_basin["sim"]).alias("sim")
                #     )
                # else:
                #     ensemble_data[basin]["df"] = df_basin

                # Compute and collect metrics for this model and basin
                metrics_dict = compute_metrics(metrics, obs_series, sim_series, date_series)
                per_model_rows.append({"basin": basin, **metrics_dict})

            # Store per-model metrics DataFrame
            results[day_key][model_name] = pl.DataFrame(per_model_rows)

        # Compute ensemble metrics for each basin
        # ensemble_rows: List[Dict] = []
        # num_models = len(results[day_key])
        # for basin, data in ensemble_data.items():
        #     df_ensemble = data["df"]
        #     obs = df_ensemble["obs"].to_numpy()
        #     # Average the summed simulations
        #     sim = (df_ensemble["sim"].to_numpy() / num_models)
        #     # Filter invalid observations
        #     valid_mask = obs >= 0
        #     obs_clean = obs[valid_mask]
        #     sim_clean = sim[valid_mask]

        #     metrics_dict = compute_metrics(metrics, obs_clean, sim_clean)
        #     ensemble_rows.append({"basin": basin, **metrics_dict})

        # results[day_key]["ensemble"] = pl.DataFrame(ensemble_rows)
    return results


def metrics_to_latex(
        df_loss_mqloss: pl.DataFrame,
        df_loss_mse: pl.DataFrame, 
        metrics_labels: List[str],
        model_labels: List[str],
        aggregation: str,
        file_name: str) -> None:
    """
    Generate a LaTeX table of performance metrics for multiple models across different forecast horizons and write the table to a .tex file.

    This function iterates over specified metrics and timesteps, computes the mean and standard deviation
    for each model and metric, and formats the results into a LaTeX table.

    Parameters
    ----------
        df (pl.DataFrame): A Polars DataFrame containing the df evaluation results.
            It should be structured such that df[horizon][model][metric] returns a series
            of metric values for the given forecast horizon and model.
        metrics_labels (List[str]): A list of metric names to include in the table (e.g., ["NSE", "KGE"]).
        model_labels (List[str]): A list of model identifiers matching the DataFrame keys
            (e.g., ["LSTM", "TCN", "PatchTST"]).
        file_name (str): The name of the output file where the LaTeX table will be saved.

    Returns
    -------
        None: Writes a LaTeX-formatted table to "file_name.tex".
    """
    
    timesteps = list(df_loss_mqloss.keys())
    timesteps_labels = {f"day{h}": f"Day {h}" for h in range(1, len(timesteps) + 1)}

    # Initialize lines for LaTeX table structure
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Metrics}",
        r"\begin{threeparttable}",
        r"\label{tab:metrics}",
        r"\begin{tabular}{ll*{4}{cc}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Time}}",
        r"& \multicolumn{2}{c}{\textbf{LSTM}}",
        r"& \multicolumn{2}{c}{\textbf{TCN}}",
        r"& \multicolumn{2}{c}{\textbf{PatchTST}}\\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
        r"& & \textbf{MQLoss} & \textbf{MSELoss}",
        r"& \textbf{MQLoss} & \textbf{MSELoss}",
        r"& \textbf{MQLoss} & \textbf{MSELoss}\\"
        r"\midrule"
    ]

    # Populate table rows with mean and std for each metric at each timestep
    for metric_label in metrics_labels:
        # Add multirow entry for current metric
        lines.append(rf"\multirow{{3}}{{*}}{{{metric_label}\tnote{{{chr(97 + metrics_labels.index(metric_label))}}}}}")
        for t in timesteps:
            # Compute mean and std for each model
            if aggregation == "mean":
                row_metrics = {
                    "mse": { model: df_loss_mse[t][model.lower()][metric_label].mean() for model in model_labels },
                    "mqloss": { model: df_loss_mqloss[t][model.lower()][metric_label].mean() for model in model_labels },
                    # "std":   { model: df[t][model.lower()][metric_label].std()  for model in model_labels }
                }
            else:
                row_metrics = {
                    "mse": { model: df_loss_mse[t][model.lower()][metric_label].median() for model in model_labels },
                    "mqloss": { model: df_loss_mqloss[t][model.lower()][metric_label].median() for model in model_labels },
                }
            
            # Format cells as 'mean (std)'
            cells = [fr"{row_metrics['mqloss'][model]:.2f} & {row_metrics['mse'][model]:.2f}" for model in model_labels]

            # Append the formatted row to the table
            lines.append(rf" & {timesteps_labels[t]} & {' & '.join(cells)} \\")
        # Add a midrule between metrics, except after the last one
        if metric_label != metrics_labels[-1]:
            lines.append(r"\midrule")

    # Close the table and add footnotes explaining metrics
    lines.extend([
        r"\bottomrule", 
        r"\end{tabular}",
        # r"\begin{tablenotes}",
        # r"\footnotesize",
        # r"(a) Nash–Sutcliffe efficiency: ($-\infty$,\,1]; values closer to one are desirable.",
        # r"(b) $\alpha$-NSE decomposition: (0,\,\,$\infty$); values closer to one are desirable.",
        # r"(c) $\beta$-NSE decomposition: ($-\infty$,\,\,$\infty$); values closer to zero are desirable.",
        # r"(d) KGE (top 2\% peak flow bias): ($-\infty$,\,\,$\infty$); values closer to zero are desirable.",
        # r"(e) Bias of FDC midsegment slope: ($-\infty$,\,\,$\infty$); values closer to zero are desirable.",
        # r"(f) Bias en el caudal bajo al 30\%: ($-\infty$,\,\,$\infty$); values closer to zero are desirable.",
        # r"(g) FMS mid‐segment slope bias: ($-\infty$,\,\,$\infty$); values closer to zero are desirable.",
        # r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}"
    ])

    # Write the assembled lines to the LaTeX file
    with open(f"tables/{file_name}_{aggregation}.tex", "w") as f:
        f.write("\n".join(lines))


def compare_futr_exog_to_table(
    lstm_mse_w_futr: Union[dict, pl.DataFrame],
    lstm_mse_wo_futr: Union[dict, pl.DataFrame]
) -> None:
    """
    Generate a LaTeX table comparing LSTM performance with and without future exogenous data.

    Args:
        lstm_mse_w_futr (dict or pl.DataFrame): Metrics for LSTM with future exogenous data.
        lstm_mse_wo_futr (dict or pl.DataFrame): Metrics for LSTM without future exogenous data.
    """
    timesteps = list(lstm_mse_wo_futr.keys())
    timesteps_labels = {f"day{h}": f"Day {h}" for h in range(1, len(timesteps) + 1)}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{caption}",
        r"\label{tab:table}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{LSTM} & \textbf{Time} & \textbf{Mean} & \textbf{Median}\\",
        r"\midrule"
    ]

    lines.append(rf"\multirow{{3}}{{*}}{{Without future inputs}}")
    for t in timesteps:
        lines.append(rf"& {timesteps_labels[t]} & {lstm_mse_wo_futr[t]['lstm']['NSE'].mean():.2f} & {lstm_mse_wo_futr[t]['lstm']['NSE'].median():.2f} \\")
    lines.append(r"\midrule")

    lines.append(rf"\multirow{{3}}{{*}}{{With future inputs}}")
    for t in timesteps:
        lines.append(rf"& {timesteps_labels[t]} & {lstm_mse_w_futr[t]['lstm']['NSE'].mean():.2f} & {lstm_mse_w_futr[t]['lstm']['NSE'].median():.2f} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(f"tables/comparing_futr_exog.tex", "w") as f:
        f.write("\n".join(lines))


def cohen_d_paired(x, y, bias_correction: bool = True) -> float:
    """
    Cohen's d for paired samples: mean(diff) / std(diff).
    If bias_correction=True applies small-sample correction (Hedges-like).
    """
    diffs = np.asarray(x) - np.asarray(y)
    if len(diffs) < 2:
        return 0.0
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)  # sample std
    if sd_diff == 0:
        return 0.0
    dz = mean_diff / sd_diff
    if not bias_correction:
        return dz
    n = len(diffs)
    J = 1 - (3 / (4 * n - 1))
    return dz * J  # corrected (Hedges-like)


def _compute_paired_wilcoxon_df(
    data: Dict[str, Dict[str, pl.DataFrame]],
    metric_col: str = "NSE",
    comparisons: List[Tuple[str, str, str, str]] = None
) -> pd.DataFrame:
    """
    Core: builds the DataFrame of paired Wilcoxon test results with p-value correction.
    """
    records = []

    for horizon, key1, key2, desc in comparisons or []:
        df1 = data.get(horizon, {}).get(key1)
        df2 = data.get(horizon, {}).get(key2)
        if df1 is None or df2 is None:
            continue

        # Join on 'basin'
        joined = df1.join(df2, on="basin", how="inner", suffix=f"_{key2}")

        col1 = metric_col
        col2 = f"{metric_col}_{key2}" if f"{metric_col}_{key2}" in joined.columns else metric_col

        if col1 not in joined.columns or col2 not in joined.columns:
            continue

        s1 = joined[col1].to_list()
        s2 = joined[col2].to_list()

        paired = [
            (v1, v2)
            for v1, v2 in zip(s1, s2)
            if v1 is not None and v2 is not None
               and not (isinstance(v1, float) and np.isnan(v1))
               and not (isinstance(v2, float) and np.isnan(v2))
        ]
        if not paired:
            continue
        a, b = zip(*paired)

        try:
            stat, p_raw = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", correction=False)
        except ValueError:
            stat, p_raw = np.nan, 1.0

        effect = cohen_d_paired(a, b, bias_correction=True)

        records.append({
            "horizon": horizon,
            "comparison": desc,
            "W": stat,
            "p_raw": p_raw,
            "effect_size": effect
        })

    if not records:
        return pd.DataFrame([])

    df = pd.DataFrame(records)
    # Holm correction of p-values per horizon
    df["p_adjusted"] = df.groupby("horizon")["p_raw"].transform(lambda ps: multipletests(ps, method="holm")[1])
    df["significant"] = df["p_adjusted"] < 0.05

    return df


def _format_number(value, kind="float"):
    """
    Helper to format numbers: uses scientific notation for very small or large values
    (for p-values, effect sizes, and W-statistic as appropriate).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    try:
        val = float(value)
    except Exception:
        return str(value)

    if kind == "p":
        # for p-values: use scientific if <1e-4, else fixed with 6 decimals
        if val < 1e-4 and val > 0:
            return f"{val:.2e}"
        else:
            return f"{val:.6f}"
    elif kind == "effect":
        # effect sizes: if very small or large, use scientific, else 4 decimals
        if 0 < abs(val) < 1e-3 or abs(val) >= 1e3:
            return f"{val:.2e}"
        else:
            return f"{val:.4f}"
    elif kind == "W":
        # W-statistic: similar logic, but usually moderate size
        if 0 < abs(val) < 1e-3 or abs(val) >= 1e3:
            return f"{val:.2e}"
        else:
            return f"{val:.4f}"
    else:
        return str(val)


def paired_wilcoxon_to_txt(
    data: Dict[str, Dict[str, pl.DataFrame]],
    metric_col: str = "NSE",
    comparisons: List[Tuple[str, str, str, str]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Runs the paired tests and dumps the results to a .txt file. Returns the file path.
    Numbers like small p-values are rendered in scientific notation when appropriate.
    """
    df = _compute_paired_wilcoxon_df(data, metric_col=metric_col, comparisons=comparisons)

    if df.empty:
        text = "No results obtained: no valid comparisons with sufficient data.\n"
    else:
        df_sorted = df.sort_values(["horizon", "comparison"]).reset_index(drop=True)
        lines = []
        for horizon, group in df_sorted.groupby("horizon"):
            lines.append(f"HORIZON: {horizon}")
            lines.append("-" * (9 + len(horizon)))
            for _, row in group.iterrows():
                sig_mark = "*" if row["significant"] else " "
                w_str = _format_number(row["W"], kind="W")
                p_raw_str = _format_number(row["p_raw"], kind="p")
                p_adj_str = _format_number(row["p_adjusted"], kind="p")
                effect_str = _format_number(row["effect_size"], kind="effect")
                lines.append(
                    f"{sig_mark} Comparison: {row['comparison']}\n"
                    f"    W-statistic: {w_str}\n"
                    f"    p_raw: {p_raw_str}\n"
                    f"    p_adjusted (Holm): {p_adj_str}\n"
                    f"    Effect size (paired Cohen's d, corrected): {effect_str}\n"
                    f"    Significant (alpha=0.05): {'Yes' if row['significant'] else 'No'}\n"
                )
            lines.append("")  # blank line between horizons
        text = "\n".join(lines)

    if output_path is None:
        output_path = f"statistical_tests/paired_wilcoxon_{metric_col}.txt"
    path = pathlib.Path(output_path)
    path.write_text(text, encoding="utf-8")


# Helpers to build comparisons:
def build_loss_vs_loss(horizons, models, losses):
    comps = []
    for h in horizons:
        for model in models:
            k1 = f"{model}_{losses[0]}"
            k2 = f"{model}_{losses[1]}"
            desc = f"{model} {losses[0]} vs {losses[1]}"
            comps.append((h, k1, k2, desc))
    return comps

def build_model_vs_model(horizons, loss_types, model_pairs):
    comps = []
    for h in horizons:
        for loss in loss_types:
            for m1, m2 in model_pairs:
                k1 = f"{m1}_{loss}"
                k2 = f"{m2}_{loss}"
                desc = f"{m1} vs {m2} ({loss})"
                comps.append((h, k1, k2, desc))
    return comps


def prepare_wilcoxon_futr_exog(
        lstm_mse_w_futr: Dict[str, pl.DataFrame],
        lstm_mse_wo_futr: Dict[str, pl.DataFrame]
) -> Dict[str, Dict[str, pl.DataFrame]]:
    """
    Prepare data for Wilcoxon test comparing LSTM with and without future exogenous data.
    """
    unified = {}
    for day in ["day1", "day2", "day3"]:
        unified[day] = {
            "lstm_w_futr": lstm_mse_w_futr[day]["lstm"].select(["basin", "NSE"]),
            "lstm_wo_futr": lstm_mse_wo_futr[day]["lstm"].select(["basin", "NSE"]),
        }

    comparisons = []
    for day in ["day1", "day2", "day3"]:
        comparisons.append(
            (day, "lstm_w_futr", "lstm_wo_futr", f"LSTM with vs without future exogenous data ({day})")
        )

    return unified, comparisons


def prepare_wilcoxon_all_models(
        mq_loss: Dict[str, pl.DataFrame],
        mse_loss: Dict[str, pl.DataFrame]
) -> Dict[str, Dict[str, pl.DataFrame]]:
    """
    Prepare data for Wilcoxon test comparing all models.
    """

    unified = {}
    for day in ["day1","day2","day3"]:
        unified[day] = {
            "lstm_MQLoss": mq_loss[day]["lstm"],
            "lstm_MSELoss": mse_loss[day]["lstm"],

            "tcn_MQLoss": mq_loss[day]["tcn"],
            "tcn_MSELoss": mse_loss[day]["tcn"],
            
            "patchtst_MQLoss": mq_loss[day]["patchtst"],
            "patchtst_MSELoss": mse_loss[day]["patchtst"]
        }

    horizons = ["day1", "day2", "day3"]
    models = ["lstm", "tcn", "patchtst"]
    loss_vs_loss = build_loss_vs_loss(horizons, models, ["MQLoss", "MSELoss"])
    model_pairs = [("lstm", "patchtst"), ("lstm", "tcn"), ("tcn", "patchtst")]
    model_vs_model = build_model_vs_model(horizons, ["MQLoss", "MSELoss"], model_pairs)
    all_comps = loss_vs_loss + model_vs_model

    return unified, all_comps