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
CONFIG=configs/experiment_v2.json
DATA=data/experiment_v2

run_stage() {
    case "$1" in
        data)
            "$PYTHON" -m scoperec.download
            if [[ ! -f data/processed/software/interactions.parquet || ! -f data/processed/software/items.parquet ]]; then
                "$PYTHON" -m scoperec.prepare
            fi
            if [[ ! -f "$DATA/protocol.json" || ! -f "$DATA/test.npz" ]]; then
                "$PYTHON" -m scoperec.protocol --config "$CONFIG" --output "$DATA"
            fi
            if [[ ! -f "$DATA/embeddings.json" || ! -f "$DATA/embeddings.npy" ]]; then
                "$PYTHON" -m scoperec.features --config "$CONFIG" --data "$DATA" --download --device "$DEVICE"
            fi
            if [[ ! -f artifacts/sid/sid.json || ! -f artifacts/sid/codes.npy ]]; then
                "$PYTHON" -m scoperec.semantic_ids --config "$CONFIG" --data "$DATA"
            fi
            ;;
        baseline)
            for seed in 17 29 43; do
                if [[ ! -f "artifacts/v2/baseline/seed_$seed/summary.json" || ! -f "artifacts/v2/baseline/seed_$seed/model.pt" ]]; then
                    "$PYTHON" -m scoperec.train --config "$CONFIG" --data "$DATA" \
                        --seed "$seed" --device "$DEVICE" --output "artifacts/v2/baseline/seed_$seed"
                fi
            done
            ;;
        parity)
            "$PYTHON" -m scoperec.upstream
            if [[ ! -f reports/v2/validation/selection.json ]]; then
                "$PYTHON" -m scoperec.genrecedit_parity --device "$DEVICE"
            elif [[ ! -f reports/v2/genrecedit_reference_parity.json ]]; then
                printf 'Missing parity record for locked selection; restore the original record.\n' >&2
                return 2
            fi
            ;;
        validation)
            if [[ -f reports/v2/validation/selection.json ]]; then
                printf 'Validation choices already locked; keeping existing selection.\n'
            else
                "$PYTHON" -m scoperec.genrecedit --split validation --seed 17 --device "$DEVICE" --limit 4096
                "$PYTHON" -m scoperec.compare_v2 --device "$DEVICE"
                "$PYTHON" -m scoperec.tune_confidence --device "$DEVICE"
                "$PYTHON" -m scoperec.final_v2 lock
            fi
            ;;
        test)
            for seed in 17 29 43; do
                if [[ ! -f "reports/v2/test/seed_$seed/completed.json" ]]; then
                    "$PYTHON" -m scoperec.final_v2 test --seed "$seed" --device "$DEVICE"
                fi
            done
            ;;
        diagnostics)
            for seed in 17 29 43; do
                if [[ ! -f "reports/v2/diagnostics/seed_$seed/integrity_and_latency.json" ]]; then
                    "$PYTHON" -m scoperec.diagnostics_v2 --seed "$seed" --device "$DEVICE"
                fi
            done
            ;;
        report)
            "$PYTHON" -m scoperec.report_v2
            ;;
        verify)
            "$PYTHON" -m pytest -q
            "$PYTHON" -m ruff check src scripts tests
            bash -n scripts/bootstrap_environment.sh scripts/reproduce.sh scripts/reproduce_v2.sh
            if [[ -f reports/test/seed_17/frozen.npz && -f artifacts/baseline/seed_17/model.pt ]]; then
                "$PYTHON" -m scoperec.audit
            fi
            "$PYTHON" -m scoperec.audit_v2
            ;;
        package)
            "$PYTHON" -m scoperec.package
            ;;
        *)
            printf 'Unknown stage: %s\n' "$1" >&2
            return 2
            ;;
    esac
}

if [[ "$STAGE" == all ]]; then
    for stage in data baseline parity validation test diagnostics report verify package; do
        printf '\nScopeRec v2 stage: %s\n' "$stage"
        run_stage "$stage"
    done
else
    run_stage "$STAGE"
fi