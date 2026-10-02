# Reproduce and Explore

## Install

Python 3.11/3.12 is supported. Install an appropriate PyTorch build using the
[official instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e '.[ml,dev]'
python -m pytest -q
python -m ruff check src scripts tests
```

CPU is sufficient for offline tests and final chart rendering. A CUDA GPU is
recommended for actual training. Experiments fit one RTX 4090; the three seeds
can run serially. The host-specific bootstrap uses an existing NVIDIA-image
PyTorch and `--system-site-packages`. [constraints-runtime.txt](../constraints-runtime.txt)
is a tested image constraint, not a portable complete lock or a PyPI-downloadable
private PyTorch build.

Set `SCOPE_PYTHON` to your environment's Python when it is not `.venv/bin/python`.
All workflow commands run from the repository root:

```bash
export SCOPE_PYTHON="$(command -v python)"
export SCOPE_DEVICE=cuda:0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
```

The feature-model revision is fixed; the current configuration names a mirror
because the official host was unreachable on the experimental machine. Review
[REFERENCES.md](REFERENCES.md) before choosing download endpoints.

## Fast Public Checks

```bash
python -m scoperec --help
python -m scoperec results verify
python -m scoperec results figures
python -m pytest -q
```

Final charts are reproduced from the included measured JSON without datasets,
checkpoints or network access. `results export` requires both completed local
experiments and their audits; it creates the same curated schema, never selected
validation-grid outputs.

## Main Released Benchmark

```bash
bash scripts/reproduce_official.sh all
```

Available stages: `data`, `features`, `parity`, `baseline`, `validation`, `test`,
`diagnostics`, `report`, `verify`, `package`. The workflow fills missing artifacts
and preserves completed models and locked selections. Configuration changes
under an existing lock are rejected; start in a fresh copy for a new experiment.
It runs Software only, not all paper datasets.

The ten-epoch training has 4,540 optimizer updates under a 10,000-step released
warmup and saves the last evaluated epoch. These odd settings are intentionally
preserved. Execution typos are repaired; FP32 microbatching implements effective
batch 1024 on 24GB GPUs. Each seed took about 32-33 minutes on the recorded host;
different environments need not match wall-clock or bitwise values.

After reproducing data and artifacts:

```bash
python -m scoperec recommend --device cuda:0 --request 0 --top-k 10
python -m scoperec recommend --method tiger --device cuda:0 --request 0
python -m scoperec recommend --method genrecedit_3000 --device cuda:0 --request 0
python -m scoperec recommend --disable-edit --device cuda:0 --request 0
```

`--history` accepts catalog product IDs in chronological order. The offline demo
returns invalid-path flags rather than hiding illegal released-benchmark output.
Weights and datasets are not shipped, so this demo requires actual reproduced
artifacts. It is not a ready-made live recommendation API.

## Temporal Confirmation

```bash
bash scripts/reproduce_v2.sh all
python -m scoperec recommend-temporal \
  --method scoperec_content_balanced --device cuda:0 --request 0
```

This uses the independent 2020 confirmation protocol. Do not combine its values
with the released benchmark. The original v1 research workflow remains in source
for tests and ablations but is not presented as a separate headline result.

## Code Map

| Responsibility | Owning Modules |
| --- | --- |
| Public commands | [cli.py](../src/scoperec/cli.py), [official_recommend.py](../src/scoperec/official_recommend.py) |
| Current mixture | [official_content.py](../src/scoperec/official_content.py), [content_scope.py](../src/scoperec/content_scope.py) |
| Released model and token layout | [official_model.py](../src/scoperec/official_model.py), [official_tokens.py](../src/scoperec/official_tokens.py) |
| Released data, features and training | [official_data.py](../src/scoperec/official_data.py), [official_features.py](../src/scoperec/official_features.py), [official_train.py](../src/scoperec/official_train.py) |
| Actual GenRecEdit optimizer | [genrecedit.py](../src/scoperec/genrecedit.py), [official_edits.py](../src/scoperec/official_edits.py) |
| Frozen temporal decoder | [model.py](../src/scoperec/model.py), [decoding.py](../src/scoperec/decoding.py) |
| Controlled final experiments | [official_experiment.py](../src/scoperec/official_experiment.py), [final_v2.py](../src/scoperec/final_v2.py) |
| Integrity and statistical checks | [official_audit.py](../src/scoperec/official_audit.py), [audit_v2.py](../src/scoperec/audit_v2.py), [metrics.py](../src/scoperec/metrics.py) |
| Final public data and figures | [release_results.py](../src/scoperec/release_results.py) |

The tested module paths are preserved instead of performing a cosmetic rewrite
of experiment code. Public documentation is organized around the current model;
internal experiment outputs live under ignored `reports/`, `data/` and `artifacts/`.

## Local Audit Versus Public Validation

`pytest` and `results verify` run on the clean source release. Dataset/checkpoint
audits require real locally reproduced artifacts. Use:

```bash
bash scripts/reproduce_official.sh verify
bash scripts/reproduce_v2.sh verify
```

Both replay statistics from saved outputs and check hashes; neither can certify
the paper's missing tokenizer/checkpoint choices. No command uploads a repository,
creates commits, configures a remote or publishes private artifacts.