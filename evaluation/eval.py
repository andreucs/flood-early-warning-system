from pathlib import Path
from evalutils import (
    compute_metrics_by_horizon,
    metrics_to_latex,
    compare_futr_exog_to_table,
    prepare_wilcoxon_futr_exog,
    prepare_wilcoxon_all_models,
    paired_wilcoxon_to_txt
)
from plotutils import (
    plot_cdf,
    plot_simulation
)

METRICS = ["NSE",
           "MSE",
           "Alpha-NSE",
           "Beta-NSE",
           "Pearson $r$",
           "KGE",
           "FHV",
           "FLV",
           "FMS",
           "Peak-Timing"]


MODELS = ["lstm", "tcn", "patchtst"]


mq_loss = compute_metrics_by_horizon(Path("results/mq"), METRICS, horizon_days=3)
mse_loss = compute_metrics_by_horizon(Path("results/mse/"), METRICS, horizon_days=3)

lstm_mse_w_futr = compute_metrics_by_horizon(Path("results/futr_exog/"), ["NSE"], horizon_days=3)
lstm_mse_wo_futr = {h:{"lstm":models["lstm"]} for h, models in mse_loss.items()}

metrics_to_latex(mq_loss, mse_loss, METRICS, MODELS, "median", "table_metrics")
metrics_to_latex(mq_loss, mse_loss, METRICS, MODELS, "mean", "table_metrics")
compare_futr_exog_to_table(lstm_mse_w_futr, lstm_mse_wo_futr)

data, comparisons = prepare_wilcoxon_futr_exog(
    lstm_mse_w_futr, 
    lstm_mse_wo_futr
)

paired_wilcoxon_to_txt(
    data=data, 
    metric_col="NSE", 
    comparisons=comparisons, 
    output_path="statistical_tests/paired_wilcoxon_futr_exog.txt"
    )

data, comparisons = prepare_wilcoxon_all_models(
    mq_loss, 
    mse_loss
)

for metric in METRICS:
    paired_wilcoxon_to_txt(
        data=data, 
        metric_col=metric, 
        comparisons=comparisons
    )

metrics_to_plot = ["NSE","KGE"]

for metric in metrics_to_plot:
    for day in range(1, 4):
        plot_cdf(
            metric=metric,
            day=day,
            figsize=(10,10),
            mqloss=mq_loss,
            mse_loss=mse_loss
        )

plot_simulation(
    model="patchtst",
    figsize=(18, 10)
)