# cajeta-timeseries

`dev.cajeta.timeseries` — classical time-series analysis for the cajeta
ecosystem: the **statsmodels role**, the way `dev.cajeta.ml` is the
scikit-learn role.

- **Series** — a value type over `Tensor<float64>` with a regular time
  index, lag operators, and explicit missing-value policy.
- **Decomposition** — trend/seasonal/residual, additive and multiplicative.
- **Stationarity** — ADF and KPSS (complementary nulls, made unmistakable
  at the call site), differencing and inverse differencing.
- **Diagnostics** — three-condition white-noise test with per-condition
  reporting, Ljung-Box, random-walk detection.
- **Correlograms** — ACF (Bartlett or constant bands, biased/adjusted,
  direct or FFT) and PACF (Yule-Walker, OLS, Levinson-Durbin).
- **Moving averages** — simple and weighted, plus the baseline forecaster.
- **AR / MA / ARMA / ARIMA** — least squares for AR; maximum likelihood
  through `cajeta.math.optim` for MA/ARMA; ARIMA wraps differencing and
  forecasts in the original scale.
- **Evaluation** — chronological splitting and rolling-origin backtesting
  are the **only** evaluation paths. No forecast API permits accidental
  lookahead; a shuffled split on a series produces an excellent score and
  a worthless model, so it does not exist here.

## The oracle, and its pin

**The parity target is statsmodels, not scikit-learn.** Every numeric
claim is pinned against a fixture computed by:

- **statsmodels 0.14.6**, with numpy 2.5.1 / scipy 1.18.0

(`tools/fixtures/gen_ts.py` asserts these versions before generating.)
Deliberate departures are catalogued in
[docs/DifferencesFromStatsmodels.md](docs/DifferencesFromStatsmodels.md).

## Build, test, tour

```
./run-tests.sh    # unit suite (cajeta-unit reflective @Test discovery)
./run-tour.sh     # self-checking tour
cajeta build      # emit build/archive/dev.cajeta.timeseries-<version>.cja
```

Depends on `dev.cajeta.ml` (`Metrics`, and the `Predictor` protocol where
forecasting genuinely fits it) and the stdlib's `cajeta.math`
(`linalg`, `fft`, `stats`, `optim`).
