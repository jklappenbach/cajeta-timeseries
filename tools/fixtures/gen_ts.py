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


def gen_stationarity():
    """U3 — ADF/KPSS (spec §4, §11.3).

    Three canonical inputs (saved): a seeded random walk (unit root — ADF
    must fail to reject), seeded white noise (stationary), and a trend-
    stationary series. Each adfuller/kpss config saves the full result
    vector [stat, pval, usedlag, nobs, crit1, crit5, crit10] so the cajeta
    side asserts every documented field, not just the statistic.
    """
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss

    rng = np.random.default_rng(7)
    n = 200
    rw = np.cumsum(rng.normal(0.0, 1.0, n))
    wn = np.random.default_rng(8).normal(0.0, 1.0, n)
    trend = 0.1 * np.arange(n, dtype=np.float64) \
        + np.random.default_rng(9).normal(0.0, 1.0, n)
    save("ts_stat_rw", rw)
    save("ts_stat_wn", wn)
    save("ts_stat_trend", trend)

    def adf(name, x, regression, autolag="AIC", maxlag=None):
        r = adfuller(x, maxlag=maxlag, regression=regression,
                     autolag=autolag)
        stat, pval, usedlag, nobs, crit = r[0], r[1], r[2], r[3], r[4]
        save(name, [stat, pval, float(usedlag), float(nobs),
                    crit["1%"], crit["5%"], crit["10%"]])

    adf("ts_adf_wn_c_aic", wn, "c")
    adf("ts_adf_rw_c_aic", rw, "c")
    adf("ts_adf_trend_c_aic", trend, "c")
    adf("ts_adf_wn_c_bic", wn, "c", autolag="BIC")
    adf("ts_adf_wn_c_tstat", wn, "c", autolag="t-stat")
    adf("ts_adf_trend_ct_aic", trend, "ct")
    adf("ts_adf_trend_ctt_aic", trend, "ctt")
    adf("ts_adf_rw_n_aic", rw, "n")
    adf("ts_adf_wn_c_fixed5", wn, "c", autolag=None, maxlag=5)

    def kp(name, x, regression):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, lags, crit = kpss(x, regression=regression,
                                          nlags="auto")
        save(name, [stat, pval, float(lags),
                    crit["10%"], crit["5%"], crit["2.5%"], crit["1%"]])

    kp("ts_kpss_wn_c", wn, "c")
    kp("ts_kpss_rw_c", rw, "c")
    kp("ts_kpss_trend_ct", trend, "ct")


def gen_diagnostics():
    """U4 — white-noise/random-walk diagnostics (spec §5).

    One fixture per white-noise failure mode (plan 4.1.2): an AR(1)
    series (fails autocorrelation only) and a heteroscedastic series
    (fails variance only); white noise and the trending series already
    exist from U3. Ljung-Box statistic+p per lag from acorr_ljungbox for
    the §5.3 parity claim.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    n = 200
    rng = np.random.default_rng(11)
    e = rng.normal(0.0, 1.0, n)
    ar1 = np.empty(n)
    ar1[0] = e[0]
    for i in range(1, n):
        ar1[i] = 0.7 * ar1[i - 1] + e[i]
    save("ts_diag_ar1", ar1)

    het = np.concatenate([
        np.random.default_rng(12).normal(0.0, 1.0, n // 2),
        np.random.default_rng(13).normal(0.0, 3.0, n // 2),
    ])
    save("ts_diag_hetero", het)

    wn = np.load(f"{OUT}/ts_stat_wn.npy")
    lb = acorr_ljungbox(wn, lags=10)
    save("ts_lb_wn", np.column_stack([lb["lb_stat"], lb["lb_pvalue"]]))
    lb2 = acorr_ljungbox(ar1, lags=10)
    save("ts_lb_ar1", np.column_stack([lb2["lb_stat"], lb2["lb_pvalue"]]))


def gen_correlograms():
    """U5 — ACF/PACF (spec §6).

    ACF values (biased and adjusted), Bartlett and constant confidence
    intervals at alpha 0.05, and PACF by ywm / ywadjusted / ols / ldb, all
    on the U4 AR(1) fixture. Plus AR(2) and MA(2) identification series
    (n = 1000, seeded) whose cutoff behaviour is verified HERE before the
    fixture is trusted (plan 5.1.7).
    """
    from statsmodels.tsa.stattools import acf, pacf

    ar1 = np.load(f"{OUT}/ts_diag_ar1.npy")
    r, confint = acf(ar1, nlags=20, alpha=0.05, fft=True)
    save("ts_acf_ar1", r)
    save("ts_acf_ar1_confint", confint)
    r2, confint2 = acf(ar1, nlags=20, alpha=0.05, fft=True,
                       bartlett_confint=False)
    save("ts_acf_ar1_confint_const", confint2)
    save("ts_acf_ar1_adj", acf(ar1, nlags=20, adjusted=True, fft=False))
    save("ts_pacf_ar1_ywm", pacf(ar1, nlags=20, method="ywm"))
    save("ts_pacf_ar1_ywa", pacf(ar1, nlags=20, method="ywadjusted"))
    save("ts_pacf_ar1_ols", pacf(ar1, nlags=20, method="ols"))
    save("ts_pacf_ar1_ldb", pacf(ar1, nlags=20, method="ldb"))

    n = 1000
    rng = np.random.default_rng(31)
    e = rng.normal(0.0, 1.0, n + 2)
    ar2 = np.zeros(n + 2)
    for i in range(2, n + 2):
        ar2[i] = 0.6 * ar2[i - 1] - 0.3 * ar2[i - 2] + e[i]
    ar2 = ar2[2:]
    ma2 = e[2:] + 0.7 * e[1:-1] + 0.4 * e[:-2]
    save("ts_ident_ar2", ar2)
    save("ts_ident_ma2", ma2)

    # Trust-but-verify the identification property before saving it as a
    # claim: PACF(AR2) outside the band at 1..2 and inside at 3..6; ACF
    # (MA2) outside at 1..2 and inside at 3..6 (Bartlett band).
    band = 1.959963984540054 / np.sqrt(n)
    p = pacf(ar2, nlags=6, method="ywm")
    assert all(abs(p[k]) > band for k in (1, 2)), p
    assert all(abs(p[k]) < band for k in (3, 4, 5, 6)), p
    a, ci = acf(ma2, nlags=6, alpha=0.05, fft=True)
    half = ci[:, 1] - a
    assert all(abs(a[k]) > half[k] for k in (1, 2)), a
    assert all(abs(a[k]) < half[k] for k in (3, 4, 5, 6)), a


def gen_arima_family():
    """U7/U8 — AutoReg by OLS (exact parity) and the MA/ARMA/ARIMA MLE
    family (spec §8), plus the forecast fixtures U8 pins (§9).

    AutoReg result vectors: [params..., bse..., llf, aic, bic, hqic,
    nobs]. ARIMA-family vectors: [params..., llf, aic, bic] (params in
    statsmodels order; sigma2 last). Forecasts: mean and 95% conf_int.
    """
    import warnings
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA

    ar2 = np.load(f"{OUT}/ts_ident_ar2.npy")
    ma2 = np.load(f"{OUT}/ts_ident_ma2.npy")

    def save_ar(name, res):
        save(name, np.concatenate([
            res.params, res.bse,
            [res.llf, res.aic, res.bic, res.hqic, float(res.nobs)]]))

    r = AutoReg(ar2, lags=2, trend="c").fit()
    save_ar("ts_ar_ar2_c", r)
    f = r.get_prediction(start=len(ar2), end=len(ar2) + 4)
    save("ts_ar_ar2_c_fc", f.predicted_mean)
    save("ts_ar_ar2_c_fc_ci", f.conf_int(alpha=0.05))

    save_ar("ts_ar_ar2_n", AutoReg(ar2, lags=2, trend="n").fit())
    save_ar("ts_ar_ar2_ct", AutoReg(ar2, lags=2, trend="ct").fit())
    save_ar("ts_ar_ar2_lags14", AutoReg(ar2, lags=[1, 4], trend="c").fit())

    def save_arima(name, res, h=None):
        save(name, np.concatenate([
            res.params, [res.llf, res.aic, res.bic]]))
        if h:
            g = res.get_forecast(steps=h)
            save(name + "_fc", g.predicted_mean)
            save(name + "_fc_ci", g.conf_int(alpha=0.05))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_arima("ts_arima_ma2", ARIMA(ma2, order=(0, 0, 2),
                                         trend="n").fit(), h=10)

        n = 1000
        rng = np.random.default_rng(41)
        e = rng.normal(0.0, 1.0, n + 1)
        arma11 = np.zeros(n + 1)
        for i in range(1, n + 1):
            arma11[i] = 0.5 * arma11[i - 1] + e[i] + 0.3 * e[i - 1]
        arma11 = arma11[1:]
        save("ts_arma11", arma11)
        save_arima("ts_arima_arma11", ARIMA(arma11, order=(1, 0, 1),
                                            trend="n").fit(), h=10)

        integ = 50.0 + np.cumsum(arma11)
        save("ts_arima111_series", integ)
        save_arima("ts_arima_111", ARIMA(integ, order=(1, 1, 1),
                                         trend="n").fit(), h=10)

        save_arima("ts_arima_ar2c", ARIMA(ar2, order=(2, 0, 0),
                                          trend="c").fit())


def main():
    print(f"statsmodels {statsmodels.__version__} fixtures -> {OUT}")
    gen_decompose()
    gen_stationarity()
    gen_diagnostics()
    gen_correlograms()
    gen_arima_family()


if __name__ == "__main__":
    main()
