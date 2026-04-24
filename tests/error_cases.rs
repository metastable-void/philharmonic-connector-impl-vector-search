mod helpers;

use helpers::{base_config, context, implementation};
use philharmonic_connector_impl_vector_search::{Implementation, ImplementationError};
use serde_json::json;

fn assert_invalid_request_contains(err: ImplementationError, needle: &str) {
    match err {
        ImplementationError::InvalidRequest { detail } => {
            assert!(
                detail.contains(needle),
                "expected detail to contain {needle:?}, got {detail:?}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_corpus_larger_than_max_corpus_size() {
    let impl_ = implementation();
    let config = json!({"max_corpus_size": 1, "timeout_ms": 2_000});
    let request = json!({
        "query_vector": [1.0],
        "corpus": [
            {"id": "a", "vector": [1.0]},
            {"id": "b", "vector": [1.0]}
        ],
        "top_k": 1
    });

    let err = impl_
        .execute(&config, &request, &context())
        .await
        .unwrap_err();
    assert_invalid_request_contains(err, "exceeds max_corpus_size");
}

#[tokio::test]
async fn rejects_empty_corpus() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": [1.0],
        "corpus": [],
        "top_k": 1
    });

    let err = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap_err();
    assert_invalid_request_contains(err, "corpus must contain at least one item");
}

#[tokio::test]
async fn rejects_top_k_zero() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": [1.0],
        "corpus": [{"id": "a", "vector": [1.0]}],
        "top_k": 0
    });

    let err = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap_err();
    assert_invalid_request_contains(err, "top_k must be greater than 0");
}

#[tokio::test]
async fn rejects_out_of_range_score_threshold() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": [1.0],
        "corpus": [{"id": "a", "vector": [1.0]}],
        "top_k": 1,
        "score_threshold": 2.0
    });

    let err = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap_err();
    assert_invalid_request_contains(err, "score_threshold must be within [-1.0, 1.0]");
}

#[tokio::test]
async fn rejects_vector_length_mismatch_with_item_offset() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": [1.0, 0.0, 0.0],
        "corpus": [
            {"id": "ok", "vector": [1.0, 0.0, 0.0]},
            {"id": "bad", "vector": [1.0, 0.0]}
        ],
        "top_k": 2
    });

    let err = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap_err();
    assert_eq!(
        err,
        ImplementationError::InvalidRequest {
            detail: "corpus item 1 has vector length 2, expected 3".to_owned(),
        }
    );
}

#[tokio::test]
async fn rejects_non_finite_values_in_vectors() {
    let impl_ = implementation();

    let overflowing_number_request = serde_json::from_str::<serde_json::Value>(
        r#"{
            "query_vector": [1e1000],
            "corpus": [{"id": "a", "vector": [1.0]}],
            "top_k": 1
        }"#,
    );

    match overflowing_number_request {
        Ok(request) => {
            let err = impl_
                .execute(&base_config(), &request, &context())
                .await
                .unwrap_err();

            match err {
                ImplementationError::InvalidRequest { detail } => {
                    assert!(
                        detail.contains("non-finite") || detail.contains("number out of range"),
                        "unexpected detail: {detail}"
                    );
                }
                other => panic!("expected InvalidRequest, got {other:?}"),
            }
        }
        Err(parse_error) => {
            assert!(parse_error.to_string().contains("number out of range"));
        }
    }
}

#[tokio::test]
async fn rejects_non_string_corpus_id() {
    let impl_ = implementation();
    let request = json!({
        "query_vector": [1.0],
        "corpus": [{"id": 7, "vector": [1.0]}],
        "top_k": 1
    });

    let err = impl_
        .execute(&base_config(), &request, &context())
        .await
        .unwrap_err();

    match err {
        ImplementationError::InvalidRequest { detail } => {
            assert!(detail.contains("id"));
            assert!(detail.contains("string"));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}
