## this is the pricer, the method lives in here
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

def bs_price(S0, K, r, T, vol):
    d1 = (np.log(S0 / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-(r) * T) * norm.cdf(d2)
    return call_price

def centre_and_spread(S0, r, T, vol):
    m = np.log(S0) + (r - 0.5 * vol ** 2) * T ## finds the centre of the log price distribution
    spread = vol * np.sqrt(T) ## this is the standard deviation of the log price distribution 
    return m, spread

def log_price_density(s, S0, r, T, vol):
    m, spread = centre_and_spread(S0, r, T, vol)
    return norm.pdf(s, loc=m, scale=spread) ## this is the density of the log price distribution (normal distribution)

def price_by_average(S0, K, r, T, vol):
    m, spread = centre_and_spread(S0, r, T, vol)
    lower = np.log(K)
    upper = m + 12 * spread ## pretty much infinity

    integrand = lambda s: (np.exp(s) - K) * log_price_density(s, S0, r, T, vol) ## integrate the payoff times the density, this is the payoff weighted by the probability (area under the normal curve)
    total, _ = quad(integrand, lower, upper)
    return np.exp(-r * T) * total ## discount back

## this converts prices into arrows, with deviations spread around the centre of the distribution, averaging them out spins the arrows around the origin, shrinking them as the spin (u) gets larger
def fingerprint_bs(u, S0, r, T, vol):
    m, spread = centre_and_spread(S0, r, T, vol)
    rotation = np.exp(1j * u * m)
    shrink = np.exp(-0.5 * vol ** 2 * T * u ** 2)
    return rotation * shrink
## this is now the slow way see wound_up

## this eliminates the issue of the left end of the distribution not converging to zero, by damping with a power of the strike price.
def damped_price(S0, K, r, T, vol, damping=1.5):
    return K ** damping * bs_price(S0, K, r, T, vol)

## this gives fourier transform of the damped price
## We can either find the damped price at every strike, or the damped payoff for every outcome. 
## Wind each one up across all strikes and add them all up: both give the same total. 
## We don’t know the price at every strike, but we do know the payoff for every outcome, because it’s just a formula. 
def wound_up(v, S0, r, T, vol, damping=1.5):
    discount = np.exp(-r * T)
    shifted = v - (damping + 1) * 1j
    bottom = damping ** 2 + damping - v ** 2 + 1j * (2 * damping + 1) * v ## the damping here isnt applied to prices but to damped payoffs
    return discount * fingerprint_bs(shifted, S0, r, T, vol) / bottom 
## we get the wound up damped price at the end because theyre equal

def price_one_strike(S0, K, r, T, vol, damping=1.5):
    k = np.log(K)
    slice_at = lambda v: (np.exp(-1j * v * k) * wound_up(v, S0, r, T, vol, damping)).real
    total, _ = quad(slice_at, 0, 200, limit=500)
    return undamp(total, k, damping)

## the two evenly spaced lists, linked by the grid rule
def fft_grids(S0, N=4096, spin_spacing=0.25):
    strike_spacing = 2 * np.pi / (N * spin_spacing)
    v = spin_spacing * np.arange(N)
    k_start = np.log(S0) - N * strike_spacing / 2  ## puts S0 in the middle
    k = k_start + strike_spacing * np.arange(N)
    return v, k

## turn the summed totals back into call prices: the 1/pi, then undamp
def undamp(total, k, damping=1.5):
    return np.exp(-damping * k) / np.pi * total

def price_all_strikes(S0, r, T, vol, damping=1.5, N=4096, spin_spacing=0.25):
    v, k = fft_grids(S0, N, spin_spacing)

    start_turn = np.exp(-1j * v * k[0])  ## the turn for the starting strike, which the FFT doesn't do
    widths = np.full(N, spin_spacing)    ## each slice counts for one strip width
    widths[0] = spin_spacing / 2         ## trapezium rule: the first edge counts half

    slices = start_turn * wound_up(v, S0, r, T, vol, damping) * widths
    totals = np.fft.fft(slices).real     ## the sum for every strike at once
    return np.exp(k), undamp(totals, k, damping)