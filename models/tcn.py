from neuralforecast import NeuralForecast
from neuralforecast.models import TCN
from neuralforecast.losses.pytorch import MSE, MQLoss
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

# Define parameters for the TCN model
nf = NeuralForecast(
    models=[
        TCN(h=3,
            input_size=365,
            loss= (
                MQLoss(level=[90]) if config.LOSS == "mqloss" else MSE()
                ),
            learning_rate=5e-4,
            kernel_size=3,
            dilations=[1,2,4,8,16],
            encoder_hidden_size=128,
            context_size=30,
            decoder_hidden_size=128,
            decoder_layers=2,
            max_steps=2_000,
            num_lr_decays=2,
            scaler_type='robust',
            early_stop_patience_steps=6,
            val_check_steps=50,
            batch_size=256,
            logger=False,
            random_seed=42
        )
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
    model_name="TCN",
    filename="tcn.parquet",
    results_dir=config.RESULTS_DIR / "mq" if config.LOSS == "mqloss" else config.RESULTS_DIR / "mse"
)