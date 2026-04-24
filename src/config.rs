//! Configuration model for `vector_search`.

/// Top-level configuration payload for the `vector_search` implementation.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSearchConfig {
    /// Maximum allowed corpus length per request.
    pub max_corpus_size: usize,
    /// End-to-end timeout budget in milliseconds for vector validation and scoring.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    2_000
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialize_rejects_unknown_fields() {
        let value = json!({
            "max_corpus_size": 10,
            "timeout_ms": 500,
            "extra": true
        });

        let err = serde_json::from_value::<VectorSearchConfig>(value).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn deserialize_requires_max_corpus_size() {
        let value = json!({"timeout_ms": 500});

        let err = serde_json::from_value::<VectorSearchConfig>(value).unwrap_err();
        assert!(err.to_string().contains("max_corpus_size"));
    }

    #[test]
    fn deserialize_defaults_timeout_ms() {
        let value = json!({"max_corpus_size": 1_000});

        let config = serde_json::from_value::<VectorSearchConfig>(value).unwrap();
        assert_eq!(config.timeout_ms, 2_000);
    }
}
