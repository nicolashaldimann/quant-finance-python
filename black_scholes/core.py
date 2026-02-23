import numpy as np
from black_scholes_model.utils import as_float_array, to_scalar_if_0d, normal_cdf

#d1 parameter definition :
def d1(S, K, T, r, sigma):
    S = as_float_array(S);
    K = as_float_array(K);
    T = as_float_array(T)
    r = as_float_array(r);
    sigma = as_float_array(sigma)
    return(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))


#d2 parameter definition:
def d2(S, K, T, r, sigma):
    S = as_float_array(S);
    K = as_float_array(K);
    T = as_float_array(T)
    r = as_float_array(r);
    sigma = as_float_array(sigma)
    return(d1(S, K, T, r, sigma) - sigma * np.sqrt(T))

#call price with black  scholes m
def call_price(S, K, T, r, sigma):
    S = as_float_array(S);
    K = as_float_array(K);
    T = as_float_array(T)
    r = as_float_array(r);
    sigma = as_float_array(sigma)
    payoff = np.maximum(S-K, 0.0)
    out = np.asarray(payoff, dtype=float)
    mask = T>0 #boleen to check that we don't divide per 0
    if np.any(mask):
        d_1 = d1(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        d_2 = d2(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        out[mask]= S[mask]*normal_cdf(d_1)-K[mask]*np.exp(-r[mask]*T[mask])*normal_cdf(d_2)

    return to_scalar_if_0d(out)

#put price with black  scholes m
def put_price(S, K, T, r, sigma):
    S = as_float_array(S);
    K = as_float_array(K);
    T = as_float_array(T)
    r = as_float_array(r);
    sigma = as_float_array(sigma)

    payoff = np.maximum(K-S,0.0)
    out = np.asarray(payoff, dtype=float)
    mask = T>0

    if np.any(mask):
        d_1 = d1(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        d_2 = d2(S[mask], K[mask], T[mask], r[mask], sigma[mask])
        out[mask] = K[mask]*np.exp(-r[mask]*T[mask])*normal_cdf(-d_2)-S[mask]*normal_cdf(-d_1)

    return to_scalar_if_0d(out)







