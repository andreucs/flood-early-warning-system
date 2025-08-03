from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
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

# Define parameters for the LSTM model
nf = NeuralForecast(
    models=[LSTM(h=3, 
                 input_size=365,
                 loss=(
                    MQLoss(level=[90]) if config.LOSS == "mqloss" else MSE()
                 ),
                 scaler_type='robust',
                 encoder_n_layers=2,
                 encoder_dropout=0.2,
                 encoder_hidden_size=128,
                 decoder_hidden_size=128,
                 decoder_layers=1,
                 learning_rate=5e-4,
                 num_lr_decays=2,
                 batch_size=256,
                 val_check_steps=50,
                 max_steps=3_000,
                 early_stop_patience_steps=8,
                 stat_exog_list= config.STATIC_INPUTS,
                 hist_exog_list= config.HISTORICAL_INPUTS,
                 # futr_exog_list=config.HISTORICAL_INPUTS,
                 recurrent=True,
                 h_train=1,
                 logger=False,
                 random_seed=42
                 )
    ],
    freq='1d' # 1 day frequency
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
    model_name="LSTM",
    filename="lstm.parquet",
    results_dir=config.RESULTS_DIR / "mq" if config.LOSS == "mqloss" else config.RESULTS_DIR / "mse"
    # results_dir=config.RESULTS_DIR / "futr_exog"
)