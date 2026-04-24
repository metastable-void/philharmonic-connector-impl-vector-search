mod helpers;

use helpers::{base_config, context, cosine_score, implementation};
use philharmonic_connector_impl_vector_search::{Implementation, VectorSearchResponse};
use serde_json::json;

#[tokio::test]
async fn large_corpus_top_k_matches_naive_full_sort() {
    let impl_ = implementation();

    let query = vec![1.0_f32, 0.0_f32, 0.5_f32, 0.25_f32];

    let mut corpus = Vec::with_capacity(2_000);
    for idx in 0..2_000_usize {
        let angle = (idx as f32) * 0.017;
        corpus.push(json!({
            "id": format!("item-{idx}"),
            "vector": [
                angle.cos(),
                angle.sin(),
                (idx % 13) as f32 / 13.0,
                (idx % 7) as f32 / 7.0
            ]
        }));
    }

    let request = json!({
        "query_vector": query,
        "corpus": corpus,
        "top_k": 20
    });

    let response = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap();
    let response: VectorSearchResponse = serde_json::from_value(response).unwrap();

    assert_eq!(response.results.len(), 20);

    for pair in response.results.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }

    let request_corpus = request["corpus"].as_array().unwrap();
    let query = request["query_vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();

    let mut naive = request_corpus
        .iter()
        .enumerate()
        .map(|item| {
            let (index, item) = item;
            let id = item["id"].as_str().unwrap().to_owned();
            let vector = item["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            (index, id, cosine_score(&query, &vector))
        })
        .collect::<Vec<_>>();

    naive.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    naive.truncate(20);

    let response_ids = response
        .results
        .iter()
        .map(|item| item.id.clone())
        .collect::<Vec<_>>();
    let naive_ids = naive
        .iter()
        .map(|(_, id, _)| id.clone())
        .collect::<Vec<_>>();

    assert_eq!(response_ids, naive_ids);
}
