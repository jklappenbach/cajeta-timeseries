# Differences from statsmodels

The oracle for `dev.cajeta.timeseries` is **statsmodels 0.14.6**
(numpy 2.5.1 / scipy 1.18.0) — see the README for the pin, and
`tools/fixtures/gen_ts.py` for the generator that asserts it. This page
mirrors `cajeta-ml`'s `DifferencesFromSklearn.md`: every *deliberate*
departure from the oracle is recorded here, one bullet per decision, as
they are made. An empty section means "verbatim parity so far".

## Series representation (spec §2)

- **Missing values are IEEE NaN with a stated per-operation policy.**
  statsmodels leans on pandas' `NaN`/mask machinery; here a `Series`
  carries `float64` NaN and every operation documents whether it
  propagates, drops, or rejects. Nothing imputes silently.
- **Irregular indexes are rejected at construction.** pandas lets an
  irregular `DatetimeIndex` through and statsmodels warns (or infers a
  frequency) later; a `Series` here refuses construction, because §1.5.4
  defines the domain as regular-interval measurements and every algorithm
  downstream assumes it.
- **There is no shuffled split for a series, anywhere.** Not a departure
  from statsmodels so much as from the `dev.cajeta.ml` conventions:
  `trainTestSplit`'s shuffle is deliberately unreachable from this
  library's types (spec §11.5).

## Correlograms (spec §6)

- **The FFT autocovariance path rides the stdlib's float32 transform**
  (`cajeta.math.fft` has no float64 FFT yet), so `Acf.compute(...,
  useFft: true)` agrees with the direct float64 path to ~1e-5 rather
  than 1e-15. The direct path is the default and is what the parity
  fixtures pin; the §6.5 same-result test asserts the two paths agree to
  1e-4. When the stdlib grows a float64 FFT the tolerance tightens for
  free.
- **PACF's default here is `"ywm"`** (spec §6.7's naming); statsmodels'
  own default is `"ywadjusted"`, which is also available and pinned.
- **`chi2` and the normal quantile are internal transliterations** —
  `cajeta.math.stats` lacks `gammainc`/`chi2Cdf` and an inverse normal
  CDF; both are implemented package-private (Ljung-Box p-values,
  confidence bands) and are registered as stdlib gaps.

## The ARIMA family (spec §8)

- **AutoReg is exact parity** (1e-8): same OLS, same MLE standard
  errors (σ² = SSR/n), same llf/AIC/BIC/HQIC definitions, same 1-based
  time-trend column.
- **MA/ARMA/ARIMA agree to optimizer tolerance, not bit-exactness.**
  Both sides maximize the same exact Kalman-filter Gaussian likelihood
  with the same Monahan stationarity/invertibility transform, but
  statsmodels drives it with scipy's L-BFGS and cajeta with
  `cajeta.math.optim`'s Nelder-Mead from a Hannan-Rissanen start —
  different optimizers stop at slightly different points on the same
  surface. Fixtures pin coefficients to 5e-3 and log-likelihood to
  1e-2; truth-recovery assertions (the oracle-independent check) guard
  the rest.
- **`Arima`'s trend surface is `"n"`/`"c"`** (const = the process mean
  μ, statsmodels' regression-with-ARMA-errors reading); `"ct"` on the
  ARMA family has no consumer yet and is deferred.
