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
