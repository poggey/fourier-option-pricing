## this tests the pricer against known results
from pricing import bs_price, fingerprint_bs

def test_bs_price_known_value():
    assert abs(bs_price(100, 100, 0.04, 1.0, 0.20) - 9.925054) < 1e-6

def test_fingerprint_at_zero_is_one():
    assert abs(fingerprint_bs(0, 100, 0.04, 1.0, 0.20) - 1) < 1e-12