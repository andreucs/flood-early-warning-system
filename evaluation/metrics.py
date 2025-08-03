import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Dict, List

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe-Effiency
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    
    Returns
    -------
    float
        Nash-Sutcliffe-Efficiency
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If all values in the observations are equal
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    # denominator of the fraction term
    denominator = np.sum((obs - np.mean(obs))**2)

    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0:
        msg = [
            "The Nash-Sutcliffe-Efficiency coefficient is not defined ",
            "for the case, that all values in the observations are equal.",
            " Maybe you should use the Mean-Squared-Error instead."
        ]
        raise RuntimeError("".join(msg))

    # numerator of the fraction term
    numerator = np.sum((sim - obs)**2)

    # calculate the NSE
    nse_val = 1 - numerator / denominator

    return nse_val


def mse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean Squared Error
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the observations
    sim : np.ndarray
        Array containing the simulations
    
    Returns
    -------
    float
        Mean Squared Error
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    """
    obs = obs.flatten()
    sim = sim.flatten()
    
    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")
    
    return np.mean((sim - obs) ** 2)


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean Absolute Error
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the observations
    sim : np.ndarray
        Array containing the simulations
    
    Returns
    -------
    float
        Mean Absolute Error
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    """
    obs = obs.flatten()
    sim = sim.flatten()
    
    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")
    
    return np.mean(np.abs(sim - obs))


def pearsonr(obs: np.array, sim: np.array) -> float:
    # verify inputs
    obs = obs.flatten()
    sim = sim.flatten()
    
    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    # r, _ = stats.pearsonr(obs.values, sim.values)
    r, _ = stats.pearsonr(obs, sim)
    return float(r)


def alpha_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Alpha decomposition of the NSE, see Gupta et al. 2009

    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations

    Returns
    -------
    float
        Alpha decomposition of the NSE

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    return np.std(sim) / np.std(obs)


def beta_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Beta decomposition of NSE. See Gupta et. al 2009
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations

    Returns
    -------
    float
        Beta decomposition of the NSE

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    return (np.mean(sim) - np.mean(obs)) / np.std(obs)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling–Gupta Efficiency (KGE), version of Gupta et al. (2009)
    
    KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    where
      r     = Pearson correlation coefficient between sim and obs
      alpha = std(sim) / std(obs)
      beta  = mean(sim) / mean(obs)
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the observations
    sim : np.ndarray
        Array containing the simulations
    
    Returns
    -------
    float
        Kling–Gupta Efficiency
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If std(obs) is zero or mean(obs) is zero (to avoid division by zero)
    """
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    # Compute components
    r = np.corrcoef(obs, sim)[0, 1]
    std_obs = np.std(obs)
    std_sim = np.std(sim)
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    
    if std_obs == 0:
        raise RuntimeError("Standard deviation of observations is zero; KGE undefined.")
    if mean_obs == 0:
        raise RuntimeError("Mean of observations is zero; KGE undefined.")
    
    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs

    # Kling–Gupta Efficiency
    kge_val = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_val


def fms(obs: np.ndarray, sim: np.ndarray, m1: float = 0.2, m2: float = 0.7) -> float:
    """[summary]
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    m1 : float, optional
        Lower bound of the middle section. Has to be in range(0,1), by default 0.2
    m2 : float, optional
        Upper bound of the middle section. Has to be in range(0,1), by default 0.2
    
    Returns
    -------
    float
        Bias of the middle slope of the flow duration curve (Yilmaz 2018).
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `m1` is not in range(0,1)
    RuntimeError
        If `m2` is not in range(0,1)
    RuntimeError
        If `m1` >= `m2`
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (m1 <= 0) or (m1 >= 1):
        raise RuntimeError("m1 has to be in the range (0,1)")

    if (m2 <= 0) or (m2 >= 1):
        raise RuntimeError("m1 has to be in the range (0,1)")

    if m1 >= m2:
        raise RuntimeError("m1 has to be smaller than m2")

    # for numerical reasons change 0s to 1e-6
    sim[sim == 0] = 1e-6
    obs[obs == 0] = 1e-6

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # calculate fms part by part
    qsm1 = np.log(sim[np.round(m1 * len(sim)).astype(int)] + 1e-6)
    qsm2 = np.log(sim[np.round(m2 * len(sim)).astype(int)] + 1e-6)
    qom1 = np.log(obs[np.round(m1 * len(obs)).astype(int)] + 1e-6)
    qom2 = np.log(obs[np.round(m2 * len(obs)).astype(int)] + 1e-6)

    fms = ((qsm1 - qsm2) - (qom1 - qom2)) / (qom1 - qom2 + 1e-6)

    return fms * 100


def fhv(obs: np.ndarray, sim: np.ndarray, h: float = 0.02) -> float:
    """Peak flow bias of the flow duration curve (Yilmaz 2018).
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    h : float, optional
        Fraction of the flows considered as peak flows. Has to be in range(0,1), by default 0.02
    
    Returns
    -------
    float
        Bias of the peak flows
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `h` is not in range(0,1)
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (h <= 0) or (h >= 1):
        raise RuntimeError("h has to be in the range (0,1)")

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)

    return fhv * 100


def flv(obs: np.ndarray, sim: np.ndarray, l: float = 0.7) -> float:
    """[summary]
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    l : float, optional
        Upper limit of the flow duration curve. E.g. 0.7 means the bottom 30% of the flows are 
        considered as low flows, by default 0.7
    
    Returns
    -------
    float
        Bias of the low flows.
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `l` is not in the range(0,1)
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (l <= 0) or (l >= 1):
        raise RuntimeError("l has to be in the range (0,1)")

    # for numerical reasons change 0s to 1e-6
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[np.round(l * len(obs)).astype(int):]
    sim = sim[np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs + 1e-6)
    sim = np.log(sim + 1e-6)
    
    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100


def mean_peak_timing(vals_obs: np.ndarray,
                     vals_sim: np.ndarray,
                     dates: np.ndarray,
                     window: int = 3,
                     resolution: str = 'D') -> float:
    """
    df debe tener columnas:
      - 'date' (Datetime)
      - 'obs'  (Float)
      - 'sim'  (Float)
    """
    
    peaks, _ = signal.find_peaks(vals_obs, distance=50, height=(None, None))

    timing_errors = []
    for idx in peaks:
        if idx - window < 0 or idx + window >= vals_obs.size:
            continue

        # buscamos pico en sim
        if vals_sim[idx] > vals_sim[idx-1] and vals_sim[idx] > vals_sim[idx+1]:
            sim_idx = idx
        else:
            window_vals = vals_sim[idx-window : idx+window+1]
            sim_idx = idx - window + int(np.argmax(window_vals))

        # fechas correspondientes
        date_obs = dates[idx]
        date_sim = dates[sim_idx]

        # calculamos delta en la resolución deseada
        delta = np.abs(
            (date_obs - date_sim).astype(f'timedelta64[{resolution}]')
        ) / np.timedelta64(1, resolution)
        timing_errors.append(delta)

    return float(np.mean(timing_errors))


def get_func_metrics() -> Dict[str, callable]:
    """Get a dictionary of metric functions
    
    Returns
    -------
    Dict[str, callable]
        Dictionary with metric names as keys and functions as values
    """
    return {
        "NSE": nse,
        "MSE": mse,
        "MAE": mae,
        "Alpha-NSE": alpha_nse,
        "Beta-NSE": beta_nse,
        "Pearson $r$": pearsonr,
        "KGE": kge,
        "FHV": fhv, 
        "FLV": flv, 
        "FMS": fms,
        "Peak-Timing": mean_peak_timing
    }


def compute_metrics(metrics: List[str], 
                    obs: np.ndarray, 
                    sim: np.ndarray,
                    dates: np.ndarray) -> Dict[str, float]:
    """Compute specified metrics between observed and simulated data
    
    Parameters
    ----------
    metrics : List[str]
        List of metric names to compute
    obs : np.ndarray
        Array containing the observed values
    sim : np.ndarray
        Array containing the simulated values
    
    Returns
    -------
    Dict[str, float]
        Dictionary with metric names as keys and computed values as values
    """
    func_metrics = get_func_metrics()
    res = {}
    
    for metric in metrics:
        if metric == "Peak-Timing":
            res[metric] = mean_peak_timing(obs, sim, dates)
        elif metric in func_metrics:
            res[metric] = func_metrics[metric](obs, sim)
        else:
            raise ValueError(f"Metric '{metric}' is not recognized.")
    
    return res


