use super::{MemoryBackend, MemoryObject, MemorySpaceFile, RecallOptions};

#[derive(Default)]
pub struct BasicBackend;

impl MemoryBackend for BasicBackend {
    fn id(&self) -> &'static str {
        "basic"
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
        let mut scored: Vec<(f64, &MemoryObject)> = space
            .objects
            .iter()
            .filter(|obj| !opts.require_high_relevance || obj.relevance.eq_ignore_ascii_case("high"))
            .map(|obj| (score_object(obj, topic, opts), obj))
            .filter(|(s, _)| *s > 0.0)
            .collect();
        if scored.is_empty() {
            scored = space
                .objects
                .iter()
                .map(|obj| (score_object(obj, topic, opts), obj))
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
        _opts: &RecallOptions,
    ) -> Option<Vec<&'a MemoryObject>> {
        let mut matches: Vec<&MemoryObject> = space
            .objects
            .iter()
            .filter(|obj| obj.kind.eq_ignore_ascii_case(kind))
            .collect();
        matches.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Some(matches)
    }
}

pub fn score_object(obj: &MemoryObject, topic: &str, opts: &RecallOptions) -> f64 {
    let mut score = match obj.relevance.to_lowercase().as_str() {
        "high" => 5.0,
        "medium" => 2.5,
        _ => 1.0,
    };
    if obj.pinned {
        score += 4.0;
    }
    let topic_lower = topic.to_lowercase();
    if !topic_lower.is_empty() {
        if obj.path.to_lowercase().contains(&topic_lower) {
            score += 3.0;
        }
        if obj.summary_short.to_lowercase().contains(&topic_lower) {
            score += 2.0;
        }
        if obj.raw.to_lowercase().contains(&topic_lower) {
            score += 1.5;
        }
    }
    for kind in &opts.prefer_kinds {
        if obj.kind.to_lowercase() == *kind {
            score += 1.0;
        }
    }
    for tag in &opts.prefer_tags {
        if obj.tags.iter().any(|t| t.to_lowercase() == *tag) {
            score += 0.8;
        }
    }
    score
}

