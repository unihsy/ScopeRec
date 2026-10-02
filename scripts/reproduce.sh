#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HF_HOME="$PWD/.cache/huggingface"
export MPLBACKEND=Agg
PYTHON="${SCOPE_PYTHON:-$PWD/.venv/bin/python}"
DEVICE="${SCOPE_DEVICE:-cuda:0}"
STAGE="${1:-all}"

run_stage() {
    case "$1" in
        data)
            "$PYTHON" -m scoperec.download
            "$PYTHON" -m scoperec.prepare
            "$PYTHON" -m scoperec.protocol
            "$PYTHON" -m scoperec.features --download --device "$DEVICE"
            "$PYTHON" -m scoperec.semantic_ids
            ;;
        baseline)
            for seed in 17 29 43; do
                "$PYTHON" -m scoperec.train --seed "$seed" --device "$DEVICE"
            done
            ;;
        validation)
            "$PYTHON" -m scoperec.experiment validation --device "$DEVICE"
            ;;
        test)
            for seed in 17 29 43; do
                "$PYTHON" -m scoperec.experiment test --seed "$seed" --device "$DEVICE"
            done
            ;;
        diagnostics)
            for seed in 17 29 43; do
                "$PYTHON" -m scoperec.diagnostics --seed "$seed" --device "$DEVICE"
            done
            "$PYTHON" -m scoperec.benchmark --device "$DEVICE"
            ;;
        report)
            "$PYTHON" -m scoperec.report
            ;;
        verify)
            "$PYTHON" -m pytest -q
            "$PYTHON" -m ruff check src scripts tests
            "$PYTHON" -m scoperec.audit
            ;;
        *)
            printf 'Unknown stage: %s\n' "$1" >&2
            return 2
            ;;
    esac
}

if [[ "$STAGE" == all ]]; then
    for stage in data baseline validation test diagnostics report verify; do
        printf '\nScopeRec stage: %s\n' "$stage"
        run_stage "$stage"
    done
else
    run_stage "$STAGE"
fi