import numpy as np
from scipy.integrate import quad

from pricing import (
    bs_price,
    damped_price,
    log_price_density,
    price_by_average,
    fingerprint_bs,
    wound_up,
    price_one_strike,
    price_all_strikes
)

## The running example, used in every test
S0, K, r, T, vol = 100, 100, 0.04, 1.0, 0.20


# Steps 1-2: the price is an average

def test_bs_price_known_value():
    assert abs(bs_price(S0, K, r, T, vol) - 9.925054) < 1e-6


def test_average_matches_bs():
    avg = price_by_average(S0, K, r, T, vol)
    assert abs(avg - bs_price(S0, K, r, T, vol)) < 1e-6


# Step 3: the fingerprint

def test_fingerprint_at_zero_is_one():
    assert abs(fingerprint_bs(0, S0, r, T, vol) - 1) < 1e-12


def test_fingerprint_matches_direct_average():
    u = 1.0
    density = lambda s: log_price_density(s, S0, r, T, vol)
    sideways, _ = quad(lambda s: np.cos(u * s) * density(s), 3, 6.5)
    upward, _ = quad(lambda s: np.sin(u * s) * density(s), 3, 6.5)
    assert abs(fingerprint_bs(u, S0, r, T, vol) - (sideways + 1j * upward)) < 1e-6


def test_damping_pulls_left_end_to_zero():
    tiny_K = 0.0001
    assert bs_price(S0, tiny_K, r, T, vol) > 99
    assert damped_price(S0, tiny_K, r, T, vol) < 0.001

def test_wound_up_matches_direct_transform():
    v = 2.0
    f = lambda k: damped_price(S0, np.exp(k), r, T, vol)
    sideways, _ = quad(lambda k: np.cos(v * k) * f(k), -20, 12, limit=500)
    upward, _ = quad(lambda k: np.sin(v * k) * f(k), -20, 12, limit=500)
    direct = sideways + 1j * upward
    assert abs(wound_up(v, S0, r, T, vol) - direct) / abs(direct) < 1e-6

def test_one_strike_matches_bs():
    for strike in [80, 100, 120]:
        fourier = price_one_strike(S0, strike, r, T, vol)
        exact = bs_price(S0, strike, r, T, vol)
        assert abs(fourier - exact) < 1e-6


def test_all_strikes_match_bs():
    strikes, prices = price_all_strikes(S0, r, T, vol)
    near = (strikes > 80) & (strikes < 120)
    exact = bs_price(S0, strikes[near], r, T, vol)
    assert np.max(np.abs(prices[near] - exact)) < 1e-6

def test_damping_does_not_change_price():
    for strike in [80, 100, 120]:
        weak = price_one_strike(S0, strike, r, T, vol, damping=1.0)
        strong = price_one_strike(S0, strike, r, T, vol, damping=1.5)
        assert abs(weak - strong) < 1e-6