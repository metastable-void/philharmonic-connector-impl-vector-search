mod helpers;

use helpers::{base_config, context, fixture_json, implementation};
use philharmonic_connector_impl_vector_search::{Implementation, VectorSearchResponse};
use serde_json::json;

#[tokio::test]
async fn top_k_greater_than_corpus_returns_all_items() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": fixture_json("query_simple.json"),
        "corpus": fixture_json("corpus_5items.json"),
        "top_k": 50
    });

    let response = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap();
    let response: VectorSearchResponse = serde_json::from_value(response).unwrap();

    assert_eq!(response.results.len(), 5);
}

#[tokio::test]
async fn top_k_equal_to_corpus_returns_all_items_in_rank_order() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": fixture_json("query_simple.json"),
        "corpus": fixture_json("corpus_5items.json"),
        "top_k": 5
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
    assert_eq!(ids, vec!["a", "b", "e", "c", "d"]);
}

#[tokio::test]
async fn large_top_k_is_handled_without_error() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": [1.0, 0.0],
        "corpus": [
            {"id": "a", "vector": [1.0, 0.0]},
            {"id": "b", "vector": [0.5, 0.5]},
            {"id": "c", "vector": [0.0, 1.0]}
        ],
        "top_k": 10_000
    });

    let response = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap();
    let response: VectorSearchResponse = serde_json::from_value(response).unwrap();

    assert_eq!(response.results.len(), 3);
}
