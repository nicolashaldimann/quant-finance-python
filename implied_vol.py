import numpy as np
from scipy.optimize import brentq

from black_scholes_model.core import call_price, put_price
from black_scholes_model.utils import as_float_array, to_scalar_if_0d


def implied_vol_call(price, S, K, T, r, sigma_min=1e-8, sigma_max=5.0):
    """
    Call implied volatility via Brent's method.
    price, S, K, T, r can be scalars or arrays (quant style).
    """
    price = as_float_array(price)
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T); r = as_float_array(r)

    out = np.full_like(S, np.nan, dtype=float)
    mask = T > 0

    # Theoretical no-arbitrage bounds to filter out impossible prices
    discK = K * np.exp(-r * T)
    lower = np.maximum(S - discK, 0.0)
    upper = S

    valid = mask & (price >= lower) & (price <= upper)

    # Solve point-by-point (Brent is not vectorized)
    idxs = np.where(valid)
    for i in zip(*idxs):
        p = float(price[i]); s = float(S[i]); k = float(K[i]); t = float(T[i]); rr = float(r[i])

        def f(sig):
            return call_price(s, k, t, rr, sig) - p

        # brentq requires a bracket where f changes sign
        try:
            out[i] = brentq(f, sigma_min, sigma_max, maxiter=200)
        except ValueError:
            out[i] = np.nan

    return to_scalar_if_0d(out)


def implied_vol_put(price, S, K, T, r, sigma_min=1e-8, sigma_max=5.0):
    """
    Put implied volatility via Brent's method.
    """
    price = as_float_array(price)
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T); r = as_float_array(r)

    out = np.full_like(S, np.nan, dtype=float)
    mask = T > 0

    discK = K * np.exp(-r * T)
    lower = np.maximum(discK - S, 0.0)
    upper = discK

    valid = mask & (price >= lower) & (price <= upper)

    idxs = np.where(valid)
    for i in zip(*idxs):
        p = float(price[i]); s = float(S[i]); k = float(K[i]); t = float(T[i]); rr = float(r[i])

        def f(sig):
            return put_price(s, k, t, rr, sig) - p

        try:
            out[i] = brentq(f, sigma_min, sigma_max, maxiter=200)
        except ValueError:
            out[i] = np.nan

    return to_scalar_if_0d(out)
