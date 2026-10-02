# References and Replication Scope

## Upstream Work

- Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER), 2023:
  [arXiv:2305.05065](https://arxiv.org/abs/2305.05065).
- Shen et al., *GenRecEdit: Adapting Model Editing for Generative Recommendation
  with Cold-Start Items*, SIGIR 2026:
  [arXiv:2603.14259v2](https://arxiv.org/html/2603.14259v2).
- Official code: [Starrylay/GenRecEdit](https://github.com/Starrylay/GenRecEdit),
  pinned commit `e6878d9c7c6e57479e840ccb8c045b11a2bd69b5`.
- Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/),
  McAuley Lab. Data rights are separate from project code.

Text models are pinned to Sentence-T5-base revision
`fc5d4628481afbbaaacd7af6bb07cf9d3865f781` for the main released benchmark and
MiniLM-L6-v2 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` for the temporal
experiment. Model cards and licenses remain authoritative for their weights.

## GenRecEdit Is Not Output LoRA

The comparison implements its actual position-wise FFN target-vector optimizer,
old-key uncentered covariance, closed-form matrix update and One-One activation.
The default layer mapping is `[0,1,2,3]`. Optimized vectors and matrix updates
were compared with executed official functions; tokenizer, unmasked search and
released metric functions were also directly executed for parity.

The port vectorizes optimization and uses a linear solve instead of forming an
inverse. Released execution typos are repaired, not taken as algorithm changes.
Fixed pseudo-sample seeds and deterministic ordering replace nondeterministic
released defaults. One shared random 400k sample is used for position moments,
and FP32 gradient accumulation fits batch 1024 within the available GPUs.

## Not Published-Table Reproduction

| Issue | Paper | Executable Defaults Used Here |
| --- | --- | --- |
| SID quantizer | Trained RQ-VAE, latent 32 | PCA-128 and Faiss residual quantization |
| Extra item token | Implementation section says none | Fourth collision token |
| Layer selection | Classifier-based localization | Fixed default mapping |
| Software preservation weight | 3000 | Shell default 1000; both reported locally |
| Metric | Recall/NDCG described as recommendation metrics | Evaluator matches first three tokens |
| Data details | Software cold fraction 24.3% | Measured release-data fraction 23.72% |

Consequently, numerical superiority over the paper is not established. The
main page compares locally executed matched models only, with exact-item metrics
alongside prefix-compatible results. No SpecGR or full-model-finetuning result
is implied by an omitted table column.

## Current ScopeRec Variant

ScopeRec-C is a content-conditional suffix mixture with a bounded confidence
budget. It differs from the original learned low-rank prefix patch. Zero new
trainable parameters does not mean zero index storage or no query-time cost.
The near-matching uniform control is preserved as a substantive limitation.

## Distribution Boundary

Pinned official source is downloaded only into local artifacts for parity,
not vendored into the source release. Model remote code is not enabled. The
mirror endpoint was necessary on the experimental host; local hashes are
fingerprints, not an independent authenticity certificate.

Do not redistribute datasets, user histories, trained weights or third-party
source under the project license. Project code uses the owner's selected
[Apache-2.0 license](../LICENSE); upstream data, models and dependencies retain
their own terms. See [PUBLISHING.md](PUBLISHING.md) for the public release boundary.