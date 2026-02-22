# Black-Scholes Model 
A lightweight, NumPy-first implementation of the Black–Scholes framework for European options:
- Closed-form pricing (Call / Put)
- Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility (Brent root-finding)
The code is written to work with scalars and NumPy arrays, with explicit handling of the maturity edge case (T = 0).

### Pricing (closed-form)
- call_price(S, K, T, r, sigma)
- put_price(S, K, T, r, sigma)
- d1(...), d2(...)

### Greeks
- Delta: delta_call, delta_put
- Gamma: gamma
- Vega: vega
- Theta: theta_call, theta_put
- Rho: rho_call, rho_put

### Implied Volatility
- implied_vol_call(price, S, K, T, r, ...)
- implied_vol_put(price, S, K, T, r, ...)

Implied vols are computed using Brent’s method (robust 1D root solver). The implementation also checks no-arbitrage bounds before attempting to solve.
### Requirements
- `numpy`
- `scipy`

Example:

```bash
pip install numpy scipy


