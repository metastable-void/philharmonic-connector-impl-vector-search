mod helpers;

use helpers::{base_config, context, fixture_json, implementation};
use philharmonic_connector_impl_vector_search::{Implementation, VectorSearchResponse};
use serde_json::json;

#[tokio::test]
async fn threshold_keeps_only_items_at_or_above_cutoff() {
    let impl_ = implementation();
    let query = fixture_json("query_simple.json");
    let corpus = fixture_json("corpus_5items.json");

    let request = json!({
        "query_vector": query,
        "corpus": corpus,
        "top_k": 5,
        "score_threshold": 0.75
    });

    let response = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap();
    let response: VectorSearchResponse = serde_json::from_value(response).unwrap();

    let ids: Vec<&str> = response
        .results
        .iter()
        .map(|item| item.id.as_str())
        .collect();
    assert_eq!(ids, vec!["a", "b"]);
}

#[tokio::test]
async fn threshold_equal_to_one_keeps_only_exact_matches() {
    let impl_ = implementation();
    let query = fixture_json("query_simple.json");
    let corpus = fixture_json("corpus_5items.json");

    let request = json!({
        "query_vector": query,
        "corpus": corpus,
        "top_k": 5,
        "score_threshold": 1.0
    });

    let response = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap();
    let response: VectorSearchResponse = serde_json::from_value(response).unwrap();
    let ids: Vec<&str> = response
        .results
        .iter()
        .map(|item| item.id.as_str())
        .collect();
    assert_eq!(ids, vec!["a"]);
}

#[tokio::test]
async fn high_valid_threshold_returns_empty_results() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": [1.0, 0.0],
        "corpus": [
            {"id": "a", "vector": [0.5, 0.8660254]},
            {"id": "b", "vector": [0.0, 1.0]}
        ],
        "top_k": 2,
        "score_threshold": 0.95
    });

    let response = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap();
    let response: VectorSearchResponse = serde_json::from_value(response).unwrap();
    assert!(response.results.is_empty());
}
