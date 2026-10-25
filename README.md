# fourier-option-pricing

# Pricing Options with the Fourier Transform

Pricing European options using the fast Fourier transform (Carr-Madan, 1999), derived from first principles in Python.

The method prices an option using only the Fourier transform of the price distribution, rather than the distribution itself. Black-Scholes is used as the test case, because its exact price is known. The same code is then applied to a model with price jumps, to show how jumps create a volatility skew.
