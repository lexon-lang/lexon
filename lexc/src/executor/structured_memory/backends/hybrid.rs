use std::collections::{HashMap, HashSet};

use super::{
    basic::score_object as basic_score, MemoryBackend, MemoryObject, MemorySpaceFile, RecallOptions,
};

/// GraphRAG / MemTree hybrid backend.
///
/// Approximates entity graphs by extracting tokens from paths, tags, and metadata,
/// then boosts memories that share rare entities with the query or with pinned/contextual nodes.
#[derive(Default)]
pub struct HybridBackend;

impl MemoryBackend for HybridBackend {
    fn id(&self) -> &'static str {
        "hybrid"
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

        let entity_freq = build_entity_freq(space);
        let topic_tokens = tokenize(topic);
        let pinned_tokens = collect_pinned_tokens(space);

        let mut scored: Vec<(f64, &MemoryObject)> = space
            .objects
            .iter()
            .filter(|obj| {
                !opts.require_high_relevance || obj.relevance.eq_ignore_ascii_case("high")
            })
            .map(|obj| {
                let base = basic_score(obj, topic, opts);
                let tokens = entity_tokens(obj);
                let graph_bonus =
                    graph_overlap_score(&tokens, &topic_tokens, &pinned_tokens, &entity_freq);
                (base + graph_bonus, obj)
            })
            .filter(|(score, _)| *score > 0.0)
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
        _opts: &RecallOptions,
    ) -> Option<Vec<&'a MemoryObject>> {
        if space.objects.is_empty() {
            return Some(Vec::new());
        }
        let entity_freq = build_entity_freq(space);
        let mut scored: Vec<(f64, &MemoryObject)> = space
            .objects
            .iter()
            .filter(|obj| obj.kind.eq_ignore_ascii_case(kind))
            .map(|obj| {
                let tokens = entity_tokens(obj);
                let centrality = tokens
                    .iter()
                    .map(|t| 1.0 / (entity_freq.get(t).cloned().unwrap_or(1) as f64))
                    .sum::<f64>();
                let recency = obj.updated_at.timestamp_millis() as f64 / 1_000_000_000f64;
                let pinned_bonus = if obj.pinned { 5.0 } else { 0.0 };
                (centrality + recency + pinned_bonus, obj)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Some(scored.into_iter().map(|(_, obj)| obj).collect())
    }
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| c.is_whitespace() || c == '/' || c == '_' || c == '-')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn entity_tokens(obj: &MemoryObject) -> Vec<String> {
    let mut tokens = Vec::new();
    tokens.extend(tokenize(&obj.path));
    tokens.extend(obj.tags.iter().map(|t| t.to_lowercase()));
    if let Some(map) = obj.metadata.as_object() {
        for value in map.values() {
            if let Some(s) = value.as_str() {
                tokens.extend(tokenize(s));
            } else if let Some(arr) = value.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        tokens.extend(tokenize(s));
                    }
                }
            }
        }
    }
    tokens
}

fn build_entity_freq(space: &MemorySpaceFile) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for obj in &space.objects {
        for token in entity_tokens(obj) {
            *freq.entry(token).or_insert(0) += 1;
        }
    }
    freq
}

fn collect_pinned_tokens(space: &MemorySpaceFile) -> HashSet<String> {
    let mut set = HashSet::new();
    for obj in &space.objects {
        if obj.pinned {
            for token in entity_tokens(obj) {
                set.insert(token);
            }
        }
    }
    set
}

fn graph_overlap_score(
    tokens: &[String],
    topic_tokens: &[String],
    pinned_tokens: &HashSet<String>,
    freq: &HashMap<String, usize>,
) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    let token_set: HashSet<&String> = tokens.iter().collect();
    for topic in topic_tokens {
        if token_set.contains(topic) {
            score += 2.5 / freq.get(topic).cloned().unwrap_or(1) as f64;
        }
    }
    for token in token_set {
        if pinned_tokens.contains(token) {
            score += 1.5;
        }
    }
    score
}
