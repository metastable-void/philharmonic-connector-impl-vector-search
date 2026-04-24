//! Search and scoring primitives for `vector_search`.

use crate::error::{Error, Result};
use crate::request::CorpusItem;
use crate::response::ResultItem;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScoredIndex {
    score: f32,
    index: usize,
}

impl Eq for ScoredIndex {}

impl Ord for ScoredIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for ScoredIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Computes cosine similarity between two vectors.
///
/// Returns `0.0` if either vector has zero norm or if vector lengths
/// differ.
pub fn cosine_score(query: &[f32], candidate: &[f32]) -> f32 {
    if query.len() != candidate.len() {
        return 0.0;
    }

    let query_norm = l2_norm(query);
    cosine_score_with_query_norm(query, query_norm, candidate)
}

/// Selects the top-k highest cosine scores from the provided corpus.
///
/// Ranking order is descending score; ties keep corpus insertion order.
pub fn rank_top_k(
    query: &[f32],
    corpus: &[CorpusItem],
    top_k: usize,
    score_threshold: Option<f32>,
) -> Vec<ResultItem> {
    if top_k == 0 || corpus.is_empty() {
        return Vec::new();
    }

    let query_norm = l2_norm(query);
    let mut heap: BinaryHeap<Reverse<ScoredIndex>> = BinaryHeap::with_capacity(top_k);

    for (index, item) in corpus.iter().enumerate() {
        let score = cosine_score_with_query_norm(query, query_norm, &item.vector);
        let candidate = ScoredIndex { score, index };

        if heap.len() < top_k {
            heap.push(Reverse(candidate));
            continue;
        }

        if let Some(worst) = heap.peek()
            && candidate > worst.0
        {
            let _ = heap.pop();
            heap.push(Reverse(candidate));
        }
    }

    let mut ranked: Vec<ScoredIndex> = heap.into_iter().map(|reverse| reverse.0).collect();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.index.cmp(&right.index))
    });

    let mut results = Vec::with_capacity(ranked.len());
    for scored in ranked {
        if let Some(threshold) = score_threshold
            && scored.score < threshold
        {
            continue;
        }

        let item = &corpus[scored.index];
        results.push(ResultItem {
            id: item.id.clone(),
            score: scored.score,
            payload: item.payload.clone(),
        });
    }

    results
}

pub(crate) fn validate_vectors(query: &[f32], corpus: &[CorpusItem]) -> Result<()> {
    validate_finite_query(query)?;

    let expected_len = query.len();
    for (item_offset, item) in corpus.iter().enumerate() {
        if item.vector.len() != expected_len {
            return Err(Error::InvalidRequest(format!(
                "corpus item {item_offset} has vector length {}, expected {expected_len}",
                item.vector.len()
            )));
        }

        validate_finite_corpus_vector(item_offset, &item.vector)?;
    }

    Ok(())
}

fn cosine_score_with_query_norm(query: &[f32], query_norm: f32, candidate: &[f32]) -> f32 {
    if query.len() != candidate.len() {
        return 0.0;
    }

    let candidate_norm = l2_norm(candidate);
    if query_norm == 0.0 || candidate_norm == 0.0 {
        return 0.0;
    }

    let dot = query
        .iter()
        .zip(candidate.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>();
    dot / (query_norm * candidate_norm)
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn validate_finite_query(query: &[f32]) -> Result<()> {
    for (value_offset, value) in query.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidRequest(format!(
                "query vector contains non-finite value at offset {value_offset}"
            )));
        }
    }
    Ok(())
}

fn validate_finite_corpus_vector(item_offset: usize, vector: &[f32]) -> Result<()> {
    for (value_offset, value) in vector.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidRequest(format!(
                "corpus item {item_offset} contains non-finite value at offset {value_offset}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use philharmonic_connector_impl_api::JsonValue;

    fn corpus_item(id: &str, vector: Vec<f32>, payload: Option<JsonValue>) -> CorpusItem {
        CorpusItem {
            id: id.to_owned(),
            vector,
            payload,
        }
    }

    fn assert_approx_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn cosine_known_values() {
        assert_approx_eq(cosine_score(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_approx_eq(cosine_score(&[1.0, 2.0], &[1.0, 2.0]), 1.0);
        assert_approx_eq(cosine_score(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);

        let sixty_degree = cosine_score(&[1.0, 0.0], &[0.5, (3.0_f32).sqrt() * 0.5]);
        assert_approx_eq(sixty_degree, 0.5);
    }

    #[test]
    fn zero_norm_vectors_score_zero() {
        assert_eq!(cosine_score(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_score(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine_score(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn top_k_ties_keep_insertion_order() {
        let query = [1.0, 0.0];
        let corpus = vec![
            corpus_item("a", vec![1.0, 0.0], None),
            corpus_item("b", vec![1.0, 0.0], None),
            corpus_item("c", vec![1.0, 0.0], None),
            corpus_item("d", vec![0.0, 1.0], None),
        ];

        let ranked = rank_top_k(&query, &corpus, 2, None);
        assert_eq!(
            ranked
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn score_threshold_drops_below_threshold_items() {
        let query = [1.0, 0.0];
        let corpus = vec![
            corpus_item("a", vec![1.0, 0.0], None),
            corpus_item("b", vec![0.8, 0.6], None),
            corpus_item("c", vec![0.0, 1.0], None),
        ];

        let ranked = rank_top_k(&query, &corpus, 3, Some(0.7));
        assert_eq!(
            ranked
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn rank_top_k_returns_empty_for_empty_corpus() {
        let ranked = rank_top_k(&[1.0, 0.0], &[], 3, None);
        assert!(ranked.is_empty());
    }

    #[test]
    fn payload_echoed_only_when_present() {
        let query = [1.0, 0.0];
        let corpus = vec![
            corpus_item(
                "with",
                vec![1.0, 0.0],
                Some(serde_json::json!({"text": "hello"})),
            ),
            corpus_item("without", vec![0.9, 0.1], None),
        ];

        let ranked = rank_top_k(&query, &corpus, 2, None);
        assert!(ranked[0].payload.is_some());
        assert!(ranked[1].payload.is_none());
    }

    #[test]
    fn validate_vectors_reports_length_mismatch_offset() {
        let query = vec![1.0, 0.0, 0.0];
        let corpus = vec![
            corpus_item("a", vec![1.0, 0.0, 0.0], None),
            corpus_item("b", vec![1.0, 0.0], None),
        ];

        let err = validate_vectors(&query, &corpus).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidRequest("corpus item 1 has vector length 2, expected 3".to_owned())
        );
    }

    #[test]
    fn validate_vectors_rejects_non_finite_query_values() {
        let query = vec![1.0, f32::NAN];
        let corpus = vec![corpus_item("a", vec![1.0, 0.0], None)];

        let err = validate_vectors(&query, &corpus).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidRequest("query vector contains non-finite value at offset 1".to_owned())
        );
    }

    #[test]
    fn validate_vectors_rejects_non_finite_corpus_values() {
        let query = vec![1.0, 0.0];
        let corpus = vec![corpus_item("a", vec![1.0, f32::INFINITY], None)];

        let err = validate_vectors(&query, &corpus).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidRequest("corpus item 0 contains non-finite value at offset 1".to_owned())
        );
    }
}
