//! Request models for `vector_search`.

use philharmonic_connector_impl_api::JsonValue;

/// One vector-search request payload.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSearchRequest {
    /// Query embedding vector used to score the corpus.
    pub query_vector: Vec<f32>,
    /// Per-request corpus of labeled vectors.
    pub corpus: Vec<CorpusItem>,
    /// Number of highest-scoring items to return.
    pub top_k: usize,
    /// Optional lower bound for accepted cosine scores.
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

/// One corpus item in a `vector_search` request.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusItem {
    /// Caller-defined stable item identifier.
    pub id: String,
    /// Candidate vector scored against `query_vector`.
    pub vector: Vec<f32>,
    /// Optional JSON payload echoed in results when this item matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<JsonValue>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialize_request_happy_path() {
        let value = json!({
            "query_vector": [1.0, 0.0],
            "corpus": [
                {"id": "a", "vector": [1.0, 0.0], "payload": {"text": "A"}},
                {"id": "b", "vector": [0.0, 1.0]}
            ],
            "top_k": 1,
            "score_threshold": 0.5
        });

        let request = serde_json::from_value::<VectorSearchRequest>(value).unwrap();
        assert_eq!(request.query_vector, vec![1.0, 0.0]);
        assert_eq!(request.corpus.len(), 2);
        assert_eq!(request.corpus[0].id, "a");
        assert_eq!(request.top_k, 1);
        assert_eq!(request.score_threshold, Some(0.5));
    }

    #[test]
    fn deserialize_rejects_missing_required_field() {
        let value = json!({
            "corpus": [{"id": "a", "vector": [1.0]}],
            "top_k": 1
        });

        let err = serde_json::from_value::<VectorSearchRequest>(value).unwrap_err();
        assert!(err.to_string().contains("query_vector"));
    }

    #[test]
    fn deserialize_rejects_wrong_top_k_type() {
        let value = json!({
            "query_vector": [1.0],
            "corpus": [{"id": "a", "vector": [1.0]}],
            "top_k": "3"
        });

        let err = serde_json::from_value::<VectorSearchRequest>(value).unwrap_err();
        assert!(err.to_string().contains("invalid type"));
    }

    #[test]
    fn deserialize_rejects_non_string_corpus_id() {
        let value = json!({
            "query_vector": [1.0],
            "corpus": [{"id": 10, "vector": [1.0]}],
            "top_k": 1
        });

        let err = serde_json::from_value::<VectorSearchRequest>(value).unwrap_err();
        assert!(err.to_string().contains("string"));
    }
}
