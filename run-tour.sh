#!/usr/bin/env bash
# Build the library .cja, compile the tour against it, run it.
# The tour is self-checking: non-zero exit means a demonstrated claim failed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
CAJETA="${CAJETA:-cajeta}"

# dev.cajeta.ml resolution: sibling checkout, else the archive run-tests.sh
# cached under build/.ml-cache (run the tests first on a bare runner).
ML_REPO="${ML_REPO:-$here/../cajeta-ml}"
ml_cja="${ML_CJA:-}"
if [[ -z "$ml_cja" && -d "$ML_REPO" ]]; then
    ( cd "$ML_REPO" && "$CAJETA" build >/dev/null )
    ml_cja="$(ls -t "$ML_REPO"/build/archive/dev.cajeta.ml-*.cja 2>/dev/null | head -1)"
fi
if [[ -z "$ml_cja" ]]; then
    ml_cja="$(ls -t "$here"/build/.ml-cache/dev.cajeta.ml-*.cja 2>/dev/null | head -1)"
fi
[[ -f "$ml_cja" ]] || { echo "could not resolve dev.cajeta.ml (run ./run-tests.sh first)" >&2; exit 1; }

echo ">> building dev.cajeta.timeseries"
"$CAJETA" build >/dev/null
art="$(ls -t "$here"/build/archive/dev.cajeta.timeseries-*.cja | head -1)"

echo ">> compiling the tour"
mkdir -p build/tour
"$CAJETA" --emit=exe --classpath="$art,$ml_cja" \
    -o build/tour/ts-tour \
    dev.cajeta.timeseries.tour.Tour.main "$here/tour/src" build/tour >/dev/null

echo ">> running"
exec ./build/tour/ts-tour
