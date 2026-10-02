# Data and Evaluation Contract

Both completed protocols use **Amazon Reviews 2023 / Software**, published by
[McAuley Lab](https://amazon-reviews-2023.github.io/). Scripts download official
archives, verify length and gzip integrity, and record SHA-256. Recorded hashes
are local fingerprints, not independently published authenticity certificates.
Raw/derived data, histories, request-level predictions and weights are excluded
from the public repository.

## Released-Code Benchmark

[official_software.json](../configs/official_software.json) aligns the executable
GenRecEdit release's `5core/timestamp_w_his` data, not all paper-text choices.

| Property | Measured Value or Rule |
| --- | --- |
| Catalog | 17,591 items |
| Test users | 3,653 |
| Training | 31,626 rows expanded to 464,196 prefix examples |
| Unique training user/history/target combinations | 31,626; released duplicates retained |
| Validation | 5,126 requests, 1,055 cold |
| Test | 9,386 requests, 2,226 cold (23.72%) |
| User filter | First half of test-row users, including first row exceeding the boundary |
| Cold definition | Target absent from filtered training-row targets |
| Text | Title, features, categories and description |
| PCA / quantizer | All-catalog white PCA; RQ on 7,011 filtered training-history items |
| Output metrics | Three-token prefix compatibility plus four-token exact item identity |
| Search | Full vocabulary; invalid paths stay in ranking |

The test-user filter, full-catalog PCA, global 5-core processing and held-out
cold-identity set are benchmark dependencies, not a claim of leak-free online
new-product discovery. A cold target may appear earlier in training histories;
37 test requests do so. Test cold request histories are not used as pseudo
supervision, but the candidate identities are shared by all adaptation methods.

The paper's 24.3% cold fraction and RQ-VAE/no-extra-token text do not match this
released-default run. Final results must retain that distinction.

## Temporal Confirmation

[experiment_v2.json](../configs/experiment_v2.json) defines a separate 0-core
global-time protocol. The base cutoff is 2018-01-01. Validation uses Jan-Jun
2018 arrivals and Jul-Dec targets. Final confirmation uses Jan-Jun 2020 arrivals,
a 2020-07-01 catalog and Jul-Dec targets. The previously inspected 2019 test
is not reused as fresh confirmation evidence.

- 300,000 hash-sampled pre-base training requests; 4,096 disjoint-user holdout.
- 12,000 final requests from 10,088 users; 707 current-batch new targets.
- Catalog 81,196 items, including 2,152 current newcomers.
- 4,762 of 68,903 otherwise sequential window requests target post-snapshot
  arrivals; exclusion is reported before group-independent sampling.
- Histories contain up to 20 strictly earlier items. Equal timestamps do not
  become each other's history. No full-period k-core or future-user filtering.
- PCA, quantizer and base fit only pre-cutoff information; old codes stay fixed.
- Legal successors are masked before normalization; invalid paths cannot enter
  final candidates in this protocol. No sampled negatives or seen-item removal.

The initial 0-core archive has 4,828,480 deduplicated interactions, 2,589,466
users and 89,246 interacted items. It is retained locally as acquisition evidence;
the public site shows only completed final comparisons.

## Shared Limitations

Review time proxies interaction time; first observed review proxies availability,
not an actual launch date. Static text and pretrained encoders lack historical
versions. Metadata excludes rating aggregates, review text and future engagement
signals, but static wording may still have temporal uncertainty. These offline
results establish no production or online A/B gain.

Download scripts do not grant data redistribution rights. Share acquisition
and processing code; review the publisher's terms before releasing any data.

Citation: Hou et al., *Bridging Language and Items for Retrieval and
Recommendation*, 2024, [arXiv:2403.03952](https://arxiv.org/abs/2403.03952).