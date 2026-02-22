import numpy as np
from black_scholes_model.core import d1, d2
from black_scholes_model.utils import as_float_array, to_scalar_if_0d, normal_cdf, normal_pdf

def delta_call(S, K, T, r, sigma):
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        out[mask] = normal_cdf(d1(S[mask], K[mask], T[mask], r[mask], sigma[mask]))

    # At maturity: delta payoff = 1|_{S>K}
    out[~mask] = (S[~mask] > K[~mask]).astype(float) #~mask means "not mask"
    return to_scalar_if_0d(out)

def delta_put(S, K, T, r, sigma):
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        out[mask] = normal_cdf(d1(S[mask], K[mask], T[mask], r[mask], sigma[mask])) - 1.0

    # à maturité: delta du payoff = -1_{S<K}
    out[~mask] = -(S[~mask] < K[~mask]).astype(float)
    return to_scalar_if_0d(out)


def gamma(S, K, T, r, sigma):
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        d_1 = d1(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        out[mask] = normal_pdf(d_1) / (S[mask] * sigma[mask] * np.sqrt(T[mask]))
    return to_scalar_if_0d(out)


def vega(S, K, T, r, sigma):
    """Vega par +1.00 de vol (pour 1 vol point: *0.01)."""
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        d_1 = d1(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        out[mask] = S[mask] * normal_pdf(d_1) * np.sqrt(T[mask])
    return to_scalar_if_0d(out)


def theta_call(S, K, T, r, sigma):
    """Theta par an (par jour: /365)."""
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        d_1 = d1(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        d_2 = d2(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        term1 = -(S[mask] * normal_pdf(d_1) * sigma[mask]) / (2.0 * np.sqrt(T[mask]))
        term2 = -r[mask] * K[mask] * np.exp(-r[mask] * T[mask]) * normal_cdf(d_2)
        out[mask] = term1 + term2
    return to_scalar_if_0d(out)


def theta_put(S, K, T, r, sigma):
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        d_1 = d1(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        d_2 = d2(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        term1 = -(S[mask] * normal_pdf(d_1) * sigma[mask]) / (2.0 * np.sqrt(T[mask]))
        term2 = +r[mask] * K[mask] * np.exp(-r[mask] * T[mask]) * normal_cdf(-d_2)
        out[mask] = term1 + term2
    return to_scalar_if_0d(out)


def rho_call(S, K, T, r, sigma):
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        d_2 = d2(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        out[mask] = K[mask] * T[mask] * np.exp(-r[mask] * T[mask]) * normal_cdf(d_2)
    return to_scalar_if_0d(out)


def rho_put(S, K, T, r, sigma):
    S = as_float_array(S); K = as_float_array(K); T = as_float_array(T)
    r = as_float_array(r); sigma = as_float_array(sigma)

    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if np.any(mask):
        d_2 = d2(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        out[mask] = -K[mask] * T[mask] * np.exp(-r[mask] * T[mask]) * normal_cdf(-d_2)
    return to_scalar_if_0d(out)




