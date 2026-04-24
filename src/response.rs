//! Response models for `vector_search`.

use philharmonic_connector_impl_api::JsonValue;

/// Vector-search response payload.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSearchResponse {
    /// Ranked nearest-neighbor results.
    pub results: Vec<ResultItem>,
}

/// One nearest-neighbor result.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResultItem {
    /// Caller-defined corpus identifier.
    pub id: String,
    /// Cosine similarity score in `[-1.0, 1.0]`.
    pub score: f32,
    /// Optional payload echoed from the matching corpus item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<JsonValue>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip_with_payload_present_and_absent() {
        let response = VectorSearchResponse {
            results: vec![
                ResultItem {
                    id: "a".to_owned(),
                    score: 0.9,
                    payload: Some(json!({"text": "alpha"})),
                },
                ResultItem {
                    id: "b".to_owned(),
                    score: 0.4,
                    payload: None,
                },
            ],
        };

        let encoded = serde_json::to_value(&response).unwrap();
        assert!(encoded["results"][0]["payload"].is_object());
        assert!(encoded["results"][1].get("payload").is_none());

        let decoded = serde_json::from_value::<VectorSearchResponse>(encoded).unwrap();
        assert_eq!(decoded, response);
    }
}
