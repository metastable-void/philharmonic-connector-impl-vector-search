#![allow(dead_code)]

use philharmonic_connector_common::{UnixMillis, Uuid};
use philharmonic_connector_impl_vector_search::{ConnectorCallContext, JsonValue, VectorSearch};

pub fn context() -> ConnectorCallContext {
    ConnectorCallContext {
        tenant_id: Uuid::nil(),
        instance_id: Uuid::nil(),
        step_seq: 0,
        config_uuid: Uuid::nil(),
        issued_at: UnixMillis(0),
        expires_at: UnixMillis(1),
    }
}

pub fn implementation() -> VectorSearch {
    VectorSearch::new()
}

pub fn base_config() -> JsonValue {
    serde_json::json!({
        "max_corpus_size": 5_000,
        "timeout_ms": 2_000
    })
}

pub fn fixture_json(path: &str) -> JsonValue {
    let full_path = format!("{}/tests/fixtures/{path}", env!("CARGO_MANIFEST_DIR"));
    let data = std::fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {full_path}: {e}"));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("failed to parse fixture {full_path}: {e}"))
}

pub fn cosine_score(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return 0.0;
    }

    let left_norm = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|v| v * v).sum::<f32>().sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }

    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();

    dot / (left_norm * right_norm)
}
