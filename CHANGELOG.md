# Changelog

All notable changes to this crate are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this crate adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-24

### Added

- Initial substantive `vector_search` implementation of the
  `philharmonic-connector-impl-api` `Implementation` trait.
- Stateless in-memory cosine kNN search over request-local corpora,
  including bounded min-heap top-k selection and optional score-threshold filtering.
- Strict request validation for corpus-size limits, `top_k`, threshold range,
  vector-length mismatches (with corpus offset), and non-finite vector values.
- Module-level unit tests plus deterministic fixture-based integration tests
  covering happy path, error cases, threshold behavior, top-k behavior, and large-corpus ranking.
