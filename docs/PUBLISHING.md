# GitHub Release Checklist

Canonical repository: [unihsy/ScopeRec](https://github.com/unihsy/ScopeRec).
Only curated source and final presentation assets belong in the public tree.
The private research workspace remains intact.

## Included

- Concise README, current architecture and Chinese diagram specification.
- Owner-supplied architecture illustration, preserved without content changes.
- Source, tests, reproducibility scripts and configuration presets.
- Two final comparison figures in PNG and SVG.
- Measured final data, per-seed values, confidence intervals and source hashes.
- Offline CI for tests, formatting rules and public result validation.

## Excluded

- Raw/processed datasets, user histories and request-level predictions.
- Checkpoints, embedding arrays, model caches and downloaded upstream code.
- Training logs, validation grids, intermediate result files and local todo.
- Reference PDFs, virtual environments, generated release directories and secrets.

Internal experiment modules are retained where they support reproduction or
tests; keeping source does not require publishing their intermediate outputs.

## Build the Local Snapshot

```bash
python -m scoperec.package --stage release/github-ready
```

This creates a source-only ZIP plus a clean upload folder. A per-file SHA-256
manifest records what was staged. Rebuilding an existing stage refuses unknown
files or locally modified copies instead of overwriting someone else's work.
Upload from the stage, not the research workspace containing private artifacts.

## Before Uploading

- Retain the owner's existing [Apache-2.0 license](../LICENSE), including it in
  the source archive and staged upload folder.
- Review dependency notices and dataset/model terms. Source-code permission
  does not authorize redistributing upstream datasets or weights.
- Review [RESULTS.md](RESULTS.md): local matched comparisons are not paper-table
  reproduction, and a small overall gain is not statistically established.
- Confirm no private data, credentials or local model artifacts are staged.
- Preserve existing repository history; do not force-push a replacement tree.

Suggested repository description:

> Frozen-backbone cold-start generative recommendation with confidence-budgeted
> semantic-ID suffix adaptation, reproducible baselines, and audited evaluation.

Suggested topics: `recommender-system`, `generative-recommendation`, `cold-start`,
`pytorch`, `semantic-id`, `model-adaptation`, `reproducible-research`.

## Result Presentation

Charts and wording emphasize measured engineering strengths without altering
numbers. Display rounding is the only numerical transformation. Absolute recall
gains are used instead of dramatic ratios from near-zero denominators. Controls,
old-item costs and uncertainty remain visible. Do not promote it as zero-memory,
zero-forgetting, production-proven or superior to the published paper.