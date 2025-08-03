# Not used in the final version of the code, but kept for reference

from neuralforecast import NeuralForecast
from neuralforecast.models import Autoformer
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
print("Loaded historical data with shape:", hist_df.shape)

static_df = load_static_attributes()
print("Loaded static attributes with shape:", static_df.shape)

if config.TRIAL:
    hist_df, static_df = set_trial(
        historical_df=hist_df,
        static_df=static_df
    )

# Define parameters for the Autoformer model
nf = NeuralForecast(
    models=[
        Autoformer(h=3,
                 input_size=256,
                 hidden_size = 64,
                 conv_hidden_size = 32,
                 dropout=0.2,
                 n_head=4,
                 loss= (
                    MQLoss(level=[90]) if config.LOSS == "mqloss" else MSE()
                 ),
                 # futr_exog_list=config.HISTORICAL_INPUTS,
                 scaler_type='robust',
                 # stat_exog_list=config.STATIC_INPUTS,
                 # hist_exog_list=config.HISTORICAL_INPUTS,
                 batch_size=256,
                 learning_rate=1e-3,
                 num_lr_decays = 2,
                 max_steps=2_000,
                 val_check_steps=50,
                 early_stop_patience_steps=5,
                 logger=False,
                 random_seed=42)
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
    model_name="Autoformer",
    filename="autoformer.parquet",
    results_dir=config.RESULTS_DIR / "mq" if config.LOSS == "mqloss" else config.RESULTS_DIR / "mse"
)