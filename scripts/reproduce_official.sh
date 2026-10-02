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
            "$PYTHON" -m scoperec.upstream --benchmark --output artifacts/official_benchmark/upstream
            "$PYTHON" -m scoperec.download --config configs/official_software.json \
                --output-dir data/raw/official_software \
                --manifest reports/official_software_download_manifest.json
            if [[ ! -f data/raw/amazon_reviews_2023/Software/meta_Software.jsonl.gz ]]; then
                "$PYTHON" -m scoperec.download
            fi
            if [[ ! -f data/official_software/protocol.json || ! -f data/official_software/test.npz ]]; then
                "$PYTHON" -m scoperec.official_data
            fi
            ;;
        features)
            if [[ ! -f artifacts/official_benchmark/sid/sid.json || ! -f artifacts/official_benchmark/sid/codes.npy ]]; then
                "$PYTHON" -m scoperec.official_features --download --device "$DEVICE"
            fi
            ;;
        parity)
            if [[ ! -f reports/official_benchmark/selection.json ]]; then
                "$PYTHON" -m scoperec.official_parity --device "$DEVICE"
            elif [[ ! -f reports/official_benchmark/parity.json ]]; then
                printf 'Restore the original parity report associated with the locked selection.\n' >&2
                return 2
            fi
            ;;
        baseline)
            for seed in 2024 2025 2026; do
                if [[ ! -f "artifacts/official_benchmark/baseline/seed_$seed/completed.json" ]]; then
                    "$PYTHON" -m scoperec.official_train --seed "$seed" --device "$DEVICE"
                fi
            done
            ;;
        validation)
            if [[ ! -f reports/official_benchmark/selection.json ]]; then
                "$PYTHON" -m scoperec.official_experiment validation --device "$DEVICE"
            else
                printf 'Official benchmark selection is locked; preserving it.\n'
            fi
            ;;
        test)
            for seed in 2024 2025 2026; do
                if [[ ! -f "reports/official_benchmark/test/seed_$seed/completed.json" ]]; then
                    "$PYTHON" -m scoperec.official_experiment test --seed "$seed" --device "$DEVICE"
                fi
            done
            ;;
        diagnostics)
            for seed in 2024 2025 2026; do
                if [[ ! -f "reports/official_benchmark/diagnostics/seed_$seed.json" ]]; then
                    "$PYTHON" -m scoperec.official_diagnostics --seed "$seed" --device "$DEVICE"
                fi
            done
            ;;
        report)
            "$PYTHON" -m scoperec.official_report
            ;;
        verify)
            "$PYTHON" -m pytest -q
            "$PYTHON" -m ruff check src scripts tests
            bash -n scripts/bootstrap_environment.sh scripts/reproduce.sh \
                scripts/reproduce_v2.sh scripts/reproduce_official.sh
            "$PYTHON" -m scoperec.official_audit
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
    for stage in data features parity baseline validation test diagnostics report verify package; do
        printf '\nOfficial Software stage: %s\n' "$stage"
        run_stage "$stage"
    done
else
    run_stage "$STAGE"
fi