– Implemented a vectorized Black–Scholes option pricer in Python to price European call/put options (NumPy/SciPy).
– Built analytical Greeks
– Developed implied volatility solver (Brent), with edge-case and no-arbitrage checks.

### Pricing (closed-form) 
call_price(S,K,T,r,sigma)
put_price(S, K, T, r, sigma)
d1(...), d2(...) 
### Greeks 
Delta: delta_call, delta_put 
Gamma: gamma 
Vega: vega 
Theta: theta_call, theta_put 
Rho: rho_call, rho_put 
### Implied Volatility 
implied_vol_call(price, S, K, T, r, ...) 
implied_vol_put(price, S, K, T, r, ...) 
Implied vols are computed using Brent’s method (robust 1D root solver). The implementation also checks no-arbitrage bounds before attempting to solve.
### Requirements
Numpy 
Scipy 

