#!/usr/bin/env python3
"""Golden-fixture generator for dev.cajeta.timeseries.

THE ORACLE IS STATSMODELS, NOT SCIKIT-LEARN. Every fixture in this
directory is computed by the pinned versions asserted below — regenerate
with the same pins or not at all (spec §10.4: an unpinned oracle makes
every fixture unreproducible).

Run:  /home/julian/code/ml/venv-sklearn-ref/bin/python gen_ts.py
Emits .npy files (C-order — np.ascontiguousarray on everything; the Npy
reader misreads fortran_order, see cajeta INDEX defect
npy-fortran-order-silent-misread) into this directory.
"""

import numpy as np
import scipy
import statsmodels

# The pin. Bump deliberately, regenerate everything, and record the bump
# in docs/DifferencesFromStatsmodels.md.
assert statsmodels.__version__ == "0.14.6", statsmodels.__version__
assert np.__version__ == "2.5.1", np.__version__
assert scipy.__version__ == "1.18.0", scipy.__version__

OUT = __file__.rsplit("/", 1)[0]


def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    np.save(f"{OUT}/{name}.npy", a)
    print(f"  {name}.npy {a.shape}")


def gen_decompose():
    """U2 — classical seasonal decomposition (spec §3).

    Three fixture sets: additive (even period 12 — the half-weight
    endpoint filter), multiplicative (same period, positive series), and
    additive with an odd period 7 (the plain ones/period filter). The
    OBSERVED series is saved too: the cajeta tests load it rather than
    re-deriving it, so libm differences cannot desynchronize the input.
    NaN endpoints in trend/resid are saved as-is — endpoint handling IS
    part of the parity claim (plan 2.1.5).
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    rng = np.random.default_rng(42)

    n, period = 48, 12
    t = np.arange(n, dtype=np.float64)
    obs = (10.0 + 0.05 * t
           + 2.0 * np.sin(2.0 * np.pi * t / period)
           + rng.normal(0.0, 0.3, n))
    r = seasonal_decompose(obs, model="additive", period=period)
    save("ts_decomp_add_observed", obs)
    save("ts_decomp_add_trend", r.trend)
    save("ts_decomp_add_seasonal", r.seasonal)
    save("ts_decomp_add_resid", r.resid)

    obs_m = ((10.0 + 0.1 * t)
             * (1.0 + 0.3 * np.sin(2.0 * np.pi * t / period))
             * (1.0 + 0.05 * rng.normal(0.0, 1.0, n)))
    assert (obs_m > 0).all()
    r = seasonal_decompose(obs_m, model="multiplicative", period=period)
    save("ts_decomp_mult_observed", obs_m)
    save("ts_decomp_mult_trend", r.trend)
    save("ts_decomp_mult_seasonal", r.seasonal)
    save("ts_decomp_mult_resid", r.resid)

    n2, period2 = 35, 7
    t2 = np.arange(n2, dtype=np.float64)
    obs_o = (5.0 + 0.1 * t2
             + 1.5 * np.sin(2.0 * np.pi * t2 / period2)
             + rng.normal(0.0, 0.2, n2))
    r = seasonal_decompose(obs_o, model="additive", period=period2)
    save("ts_decomp_odd_observed", obs_o)
    save("ts_decomp_odd_trend", r.trend)
    save("ts_decomp_odd_seasonal", r.seasonal)
    save("ts_decomp_odd_resid", r.resid)


def main():
    print(f"statsmodels {statsmodels.__version__} fixtures -> {OUT}")
    gen_decompose()


if __name__ == "__main__":
    main()
