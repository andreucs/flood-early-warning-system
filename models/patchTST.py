from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MQLoss, MSE
from pytorch_lightning import seed_everything
import logging
import config
from utils import (
    load_dynamic_timeseries,
    load_static_attributes,
    set_trial,
    save_results
)

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
seed_everything(42, workers=True)


hist_df = load_dynamic_timeseries()
static_df = load_static_attributes()

if config.TRIAL:
    hist_df, static_df = set_trial(
        historical_df=hist_df,
        static_df=static_df
    )

# Define parameters for the PatchTST model
nf = NeuralForecast(
    models=[
        PatchTST(h=3,
                 input_size=365,
                 patch_len=32,
                 stride=24,
                 dropout=0.3,
                 revin=False,
                 hidden_size=256,
                 n_heads=4,
                 scaler_type='robust',
                 loss= (
                    MQLoss(level=[90]) if config.LOSS == "mqloss" else MSE()
                 ),
                 learning_rate=1e-4,
                 num_lr_decays=2,
                 max_steps=3_000,
                 val_check_steps=50,
                 # futr_exog_list=config.HISTORICAL_INPUTS,
                 early_stop_patience_steps=8,
                 logger=False,
                 random_seed=42
        ),
    ],
    freq="1d" # 1 day frequency
)

fcst_df = nf.cross_validation(
    df=hist_df,
    static_df=static_df,
    val_size=config.VAL_SIZE,
    test_size=config.TEST_SIZE,
    step_size=1,
    n_windows=None
)

save_results(
    fcst_df=fcst_df,
    filename="patchtst.parquet",
    model_name="PatchTST",
    results_dir=config.RESULTS_DIR / "mq" if config.LOSS == "mqloss" else config.RESULTS_DIR / "mse"
)