import numpy as np
from scipy.integrate import quad

from pricing import (
    bs_price,
    log_price_density,
    price_by_average,
    fingerprint_bs,
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