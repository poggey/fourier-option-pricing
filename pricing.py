## this is the pricer, the method lives in here
import numpy as np
from scipy.stats import norm

def bs_price(S0, K, r, T, vol):
    d1 = (np.log(S0 / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-(r) * T) * norm.cdf(d2)
    return call_price

def fingerprint_bs(u, S0, r, T, vol):
    m = np.log(S0) + (r - 0.5 * vol ** 2) * T
    fingerprint_u = np.exp((1j * u * m) - (0.5 * vol ** 2 * T * u ** 2))
    return fingerprint_u
