use super::{basic::score_object as basic_score, MemoryBackend, MemoryObject, MemorySpaceFile, RecallOptions};

/// Simplified RAPTOR-style backend that clusters memories by tags and recency.
#[derive(Default)]
pub struct RaptorBackend;

impl MemoryBackend for RaptorBackend {
    fn id(&self) -> &'static str {
        "raptor"
    }

    fn order_for_topic<'a>(
        &self,
        space: &'a MemorySpaceFile,
        topic: &str,
        opts: &RecallOptions,
    ) -> Option<Vec<&'a MemoryObject>> {
        if space.objects.is_empty() {
            return Some(Vec::new());
        }
        let topic_tokens = topic
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>();
        let mut scored: Vec<(f64, &MemoryObject)> = space
            .objects
            .iter()
            .filter(|obj| !opts.require_high_relevance || obj.relevance.eq_ignore_ascii_case("high"))
            .map(|obj| {
                let base = basic_score(obj, topic, opts);
                let cluster_bonus =
                    cluster_overlap(obj, &topic_tokens).max(cluster_overlap(obj, &opts.prefer_tags));
                (base + cluster_bonus, obj)
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();
        if scored.is_empty() {
            scored = space
                .objects
                .iter()
                .map(|obj| (basic_score(obj, topic, opts), obj))
                .collect();
        }
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.updated_at.cmp(&a.1.updated_at))
        });
        Some(scored.into_iter().map(|(_, obj)| obj).collect())
    }

    fn order_for_kind<'a>(
        &self,
        space: &'a MemorySpaceFile,
        kind: &str,
        opts: &RecallOptions,
    ) -> Option<Vec<&'a MemoryObject>> {
        let mut matches: Vec<(f64, &MemoryObject)> = space
            .objects
            .iter()
            .filter(|obj| obj.kind.eq_ignore_ascii_case(kind))
            .map(|obj| {
                let recency = obj.updated_at.timestamp_millis() as f64;
                let cluster = cluster_overlap(obj, &opts.prefer_tags);
                ((recency / 1_000_000_000f64) + cluster + if obj.pinned { 5.0 } else { 0.0 }, obj)
            })
            .collect();
        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Some(matches.into_iter().map(|(_, obj)| obj).collect())
    }
}

fn cluster_overlap(obj: &MemoryObject, tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    for token in tokens {
        if obj
            .tags
            .iter()
            .any(|tag| tag.to_lowercase() == *token)
            || obj.summary_long.to_lowercase().contains(token)
        {
            score += 1.5;
        }
    }
    score
}

