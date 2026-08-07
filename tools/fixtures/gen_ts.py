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


def main():
    print(f"statsmodels {statsmodels.__version__} fixtures -> {OUT}")
    # Fixtures land unit by unit (U2 decomposition onward).


if __name__ == "__main__":
    main()
