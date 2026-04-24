//! Stateless in-memory cosine k-nearest-neighbor search connector.
//!
//! `vector_search` implements the shared
//! [`philharmonic_connector_impl_api::Implementation`] trait for
//! workloads where each request carries its own search corpus.
//! The connector does not persist vectors or call external services:
//! it validates the request, computes cosine similarity against each
//! corpus item, keeps only the top-k results via a bounded min-heap,
//! and returns ranked matches.
//!
//! This implementation targets request-local corpora in the hundreds to
//! low-thousands of items. It uses plain `[f32]` math and remains fully
//! deterministic under test.

mod config;
mod error;
mod request;
mod response;
pub mod search;

use crate::error::Error;
use std::time::Duration;

pub use crate::config::VectorSearchConfig;
pub use crate::request::{CorpusItem, VectorSearchRequest};
pub use crate::response::{ResultItem, VectorSearchResponse};
pub use philharmonic_connector_impl_api::{
    ConnectorCallContext, Implementation, ImplementationError, JsonValue, async_trait,
};

const NAME: &str = "vector_search";

/// `vector_search` connector implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct VectorSearch;

impl VectorSearch {
    /// Builds a new stateless `vector_search` implementation.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Implementation for VectorSearch {
    fn name(&self) -> &str {
        NAME
    }

    async fn execute(
        &self,
        config: &JsonValue,
        request: &JsonValue,
        _ctx: &ConnectorCallContext,
    ) -> Result<JsonValue, ImplementationError> {
        let config: VectorSearchConfig = serde_json::from_value(config.clone())
            .map_err(|e| Error::InvalidConfig(e.to_string()))
            .map_err(ImplementationError::from)?;

        if config.max_corpus_size == 0 {
            return Err(ImplementationError::from(Error::InvalidConfig(
                "max_corpus_size must be greater than 0".to_owned(),
            )));
        }

        let request: VectorSearchRequest = serde_json::from_value(request.clone())
            .map_err(|e| Error::InvalidRequest(e.to_string()))
            .map_err(ImplementationError::from)?;

        validate_request_shape(&config, &request).map_err(ImplementationError::from)?;

        let query_vector = request.query_vector.clone();
        let corpus = request.corpus.clone();
        let top_k = request.top_k;
        let score_threshold = request.score_threshold;

        let worker = tokio::task::spawn_blocking(move || {
            search::validate_vectors(&query_vector, &corpus)?;
            Ok::<_, Error>(search::rank_top_k(
                &query_vector,
                &corpus,
                top_k,
                score_threshold,
            ))
        });

        let joined = tokio::time::timeout(Duration::from_millis(config.timeout_ms), worker)
            .await
            .map_err(|_| Error::UpstreamTimeout)
            .map_err(ImplementationError::from)?;

        let scored = joined
            .map_err(|e| Error::Internal(format!("vector-search worker join failure: {e}")))
            .map_err(ImplementationError::from)?;

        let response = VectorSearchResponse {
            results: scored.map_err(ImplementationError::from)?,
        };

        serde_json::to_value(response)
            .map_err(|e| Error::Internal(e.to_string()))
            .map_err(ImplementationError::from)
    }
}

fn validate_request_shape(
    config: &VectorSearchConfig,
    request: &VectorSearchRequest,
) -> error::Result<()> {
    if request.corpus.is_empty() {
        return Err(Error::InvalidRequest(
            "corpus must contain at least one item".to_owned(),
        ));
    }

    if request.corpus.len() > config.max_corpus_size {
        return Err(Error::InvalidRequest(format!(
            "corpus length {} exceeds max_corpus_size {}",
            request.corpus.len(),
            config.max_corpus_size
        )));
    }

    if request.top_k == 0 {
        return Err(Error::InvalidRequest(
            "top_k must be greater than 0".to_owned(),
        ));
    }

    if let Some(score_threshold) = request.score_threshold
        && !(-1.0..=1.0).contains(&score_threshold)
    {
        return Err(Error::InvalidRequest(
            "score_threshold must be within [-1.0, 1.0]".to_owned(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_stable() {
        let impl_ = VectorSearch::new();
        assert_eq!(impl_.name(), "vector_search");
    }
}
