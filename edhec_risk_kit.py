import pandas as pd
import scipy.stats
import numpy as np

def drawdown(return_series: pd.Series): 
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
      The wealth index
      Previous peaks
      Percent drawdowns
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama French data set for the returns
    of the top and bottom deciles by market cap
    """
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                          header=0, 
                          index_col=0, 
                          na_values=-99.99)
    # Changing the header names
    rets = me_m[['Lo 10','Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    # Dividing the percentages to work with raw percent info
    rets = rets/100
    # Formatting of the index column
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets


def get_hfi_returns():
    """
    Load & format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv',
                          header=0, 
                          index_col=0, 
                          parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0) #don't make n-1 correction, we want population std dev.
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0) #don't make n-1 correction, we want population std dev.
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise.
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def semideviation(r):
    """
    Returns the semi-deviation aka negative semi-deviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def VaR_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(VaR_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) # negative sign because we generally report VaR as positive values.
    else: 
        raise TypeError("Expected r to be Series or DataFrame")

def VaR_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the mod VaR is returned,
    using Cornish-Fisher modification!
    """
    # Compute the z-score
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        # Modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def CVaR_historic(r, level=5):
    """
    Computes the conditional VaR of a Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -VaR_historic(r, level=level)  # Find me all the returns less than the hist VaR
        return -r[is_beyond].mean()  # Flip the sign again you want to report a positive number for VaR
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(CVaR_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")