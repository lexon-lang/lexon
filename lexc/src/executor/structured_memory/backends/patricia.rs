use super::{basic::score_object as basic_score, MemoryBackend, MemoryObject, MemorySpaceFile, RecallOptions};

#[derive(Default)]
pub struct PatriciaBackend;

impl MemoryBackend for PatriciaBackend {
    fn id(&self) -> &'static str {
        "patricia"
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
        let topic_parts = tokenize_topic(topic);
        let mut scored: Vec<(f64, &MemoryObject)> = space
            .objects
            .iter()
            .filter(|obj| !opts.require_high_relevance || obj.relevance.eq_ignore_ascii_case("high"))
            .map(|obj| {
                let base = basic_score(obj, topic, opts);
                let prefix_bonus = patricia_prefix_depth(&obj.path, &topic_parts) as f64 * 2.0;
                (base + prefix_bonus, obj)
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();
        if scored.is_empty() {
            scored = space
                .objects
                .iter()
                .map(|obj| {
                    let base = basic_score(obj, topic, opts);
                    (base, obj)
                })
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
        if space.objects.is_empty() {
            return Some(Vec::new());
        }
        let mut matches: Vec<(f64, &MemoryObject)> = space
            .objects
            .iter()
            .filter(|obj| obj.kind.eq_ignore_ascii_case(kind))
            .map(|obj| {
                let depth = obj.path.split('/').count() as f64;
                let bonus = if obj.pinned { 2.0 } else { 0.0 };
                let preference = if opts.prefer_tags.is_empty() {
                    0.0
                } else if obj
                    .tags
                    .iter()
                    .any(|t| opts.prefer_tags.contains(&t.to_lowercase()))
                {
                    1.0
                } else {
                    0.0
                };
                ((1.0 / depth) + bonus + preference, obj)
            })
            .collect();
        matches.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.updated_at.cmp(&a.1.updated_at))
        });
        Some(matches.into_iter().map(|(_, obj)| obj).collect())
    }
}

fn tokenize_topic(topic: &str) -> Vec<String> {
    topic
        .split(|c: char| c == '/' || c.is_whitespace())
        .filter(|p| !p.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn patricia_prefix_depth(path: &str, topic_parts: &[String]) -> usize {
    if topic_parts.is_empty() {
        return 0;
    }
    let path_parts: Vec<String> = path
        .split('/')
        .filter(|p| !p.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    let mut depth = 0;
    for (idx, part) in topic_parts.iter().enumerate() {
        if let Some(path_part) = path_parts.get(idx) {
            if path_part == part {
                depth += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    depth
}

