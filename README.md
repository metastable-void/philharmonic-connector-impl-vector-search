# philharmonic-connector-impl-vector-search

Part of the Philharmonic workspace: https://github.com/metastable-void/philharmonic-workspace

`philharmonic-connector-impl-vector-search` provides the `vector_search` connector implementation as a stateless in-memory cosine k-nearest-neighbor engine. Each request carries its own query vector and corpus of labeled vectors, the implementation validates schema and vector invariants, computes cosine similarity with safe zero-norm handling, selects top-k matches using a bounded min-heap, optionally applies a score threshold, and returns ranked `{id, score, payload}` results through the shared `philharmonic-connector-impl-api` contract.

## Contributing

This crate is developed as a submodule of the Philharmonic
workspace. Workspace-wide development conventions — git workflow,
script wrappers, Rust code rules, versioning, terminology — live
in the workspace meta-repo at
[metastable-void/philharmonic-workspace](https://github.com/metastable-void/philharmonic-workspace),
authoritatively in its
[`CONTRIBUTING.md`](https://github.com/metastable-void/philharmonic-workspace/blob/main/CONTRIBUTING.md).

SPDX-License-Identifier: Apache-2.0 OR MPL-2.0
