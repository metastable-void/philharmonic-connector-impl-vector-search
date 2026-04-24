mod helpers;

use helpers::{base_config, context, fixture_json, implementation};
use philharmonic_connector_impl_vector_search::{Implementation, VectorSearchResponse};

fn assert_approx_eq(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "actual={actual}, expected={expected}"
    );
}

#[tokio::test]
async fn returns_ranked_results_for_simple_fixture() {
    let impl_ = implementation();
    let config = base_config();
    let request = fixture_json("request_happy_top3.json");

    let response = impl_.execute(&config, &request, &context()).await.unwrap();
    let response: VectorSearchResponse = serde_json::from_value(response).unwrap();

    let ids: Vec<&str> = response
        .results
        .iter()
        .map(|item| item.id.as_str())
        .collect();
    assert_eq!(ids, vec!["a", "b", "e"]);

    assert_approx_eq(response.results[0].score, 1.0);
    assert_approx_eq(response.results[1].score, 0.8);
    assert_approx_eq(response.results[2].score, 0.6);

    assert!(response.results[0].payload.is_some());
    assert!(response.results[1].payload.is_none());
    assert!(response.results[2].payload.is_some());
}
