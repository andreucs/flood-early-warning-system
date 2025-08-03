from datetime import datetime
from pathlib import Path

TRIAL = True
START_DATE = "1960-01-01"
END_DATE = "2020-12-31"
START_DATE_VAL = "2000-01-01"
START_DATE_TEST = "2010-01-01"
# N_BASINS = len(BASINS)
N_BASINS = 50  # We can use a smaller number for testing
LOSS = "mse"  # Options: "mse", "mqloss"

ATTRIBUTES_DIR = "../data/CAMELS_DE/attributes"
TIMESERIES_DIR = "../data/CAMELS_DE/timeseries"
RESULTS_DIR = Path("results")
BASINS_DIR = "basins.txt"

VAL_SIZE = (
    datetime.strptime(START_DATE_TEST, "%Y-%m-%d")-datetime.strptime(START_DATE_VAL, "%Y-%m-%d")
    ).days
TEST_SIZE = (
    datetime.strptime(END_DATE, "%Y-%m-%d")-datetime.strptime(START_DATE_TEST, "%Y-%m-%d")
    ).days

STATIC_INPUTS = [
    'area', 
    'elev_mean', 
    'clay_0_30cm_mean', 
    'sand_0_30cm_mean', 
    'silt_0_30cm_mean',
    'artificial_surfaces_perc', 
    'agricultural_areas_perc', 
    'forests_and_seminatural_areas_perc',
    'wetlands_perc', 
    'water_bodies_perc', 
    'p_mean', 
    'p_seasonality', 
    'frac_snow',
    'high_prec_freq', 
    'low_prec_freq', 
    'high_prec_dur', 
    'low_prec_dur'
]

HISTORICAL_INPUTS = [
    'precipitation_mean', 
    'precipitation_stdev',
    'radiation_global_mean', 
    'temperature_min', 
    'temperature_max'
]

TARGET = 'discharge_spec_obs'