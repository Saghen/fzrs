use super::Matcher;
use super::backend::MatcherBackend;
use crate::{Match, MatchIndices};
use alloc::vec::Vec;

/// Patterns matched independently, where a haystack matches when all of the
/// non-negated patterns match and none of the negated patterns match.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(super) enum CompiledPatterns {
    Empty,
    Single(CompiledPattern),
    Multi(Vec<CompiledPattern>),
}

impl CompiledPatterns {
    pub(super) fn is_empty(&self) -> bool {
        matches!(self, CompiledPatterns::Empty)
    }
}

#[derive(Debug, Clone)]
pub(super) struct CompiledPattern {
    pub(super) negated: bool,
    pub(super) needs_unicode: bool,
    pub(super) max_typos: Option<u16>,
    pub(super) backend: MatcherBackend,
}

impl Matcher {
    pub(super) fn match_one_multi<S: AsRef<str>>(
        patterns: &mut [CompiledPattern],
        haystack: S,
        index: u32,
    ) -> Option<Match> {
        let haystack = haystack.as_ref();
        let mut combined = Match::from_index(index as usize);
        for pattern in patterns {
            let result = Self::dispatch_pattern_one(pattern, haystack, index);
            if pattern.negated {
                if result.is_some() {
                    return None;
                }
            } else {
                let m = result?;
                combined.score = combined.score.saturating_add(m.score);
                combined.exact |= m.exact;
                #[cfg(feature = "match_end_col")]
                {
                    combined.end_col = combined.end_col.max(m.end_col);
                }
            }
        }
        Some(combined)
    }

    pub(super) fn match_one_indices_multi<S: AsRef<str>>(
        patterns: &mut [CompiledPattern],
        haystack: S,
        index: u32,
    ) -> Option<MatchIndices> {
        let haystack = haystack.as_ref();
        let mut combined = MatchIndices::from_index(index as usize);
        for pattern in patterns {
            if pattern.negated {
                if Self::dispatch_pattern_one(pattern, haystack, index).is_some() {
                    return None;
                }
            } else {
                let m = Self::dispatch_pattern_one_indices(pattern, haystack, index)?;
                combined.score = combined.score.saturating_add(m.score);
                combined.exact |= m.exact;
                combined.indices.extend(m.indices);
            }
        }
        // Indices are reported in reverse order, and patterns may share matched chars.
        combined.indices.sort_unstable_by(|a, b| b.cmp(a));
        combined.indices.dedup();
        Some(combined)
    }

    /// Matches multiple patterns by matching the first non-negated pattern
    /// against every haystack, then re-matching each remaining pattern
    /// against only the haystacks that survived the previous patterns.
    /// Scores are summed across the non-negated patterns.
    pub(super) fn match_list_multi_into<S: AsRef<str>>(
        patterns: &mut [CompiledPattern],
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        let base_pattern_idx = patterns.iter().position(|p| !p.negated);
        let mut candidates = Vec::new();
        match base_pattern_idx {
            Some(i) => Self::dispatch_pattern_into(
                &mut patterns[i],
                haystacks,
                haystack_index_offset,
                &mut candidates,
            ),
            // All patterns are negated, so every haystack is a candidate.
            None => {
                let indices = (0..haystacks.len()).map(|i| i + haystack_index_offset as usize);
                candidates.extend(indices.map(Match::from_index));
            }
        }

        let mut gathered: Vec<&str> = Vec::new();
        let mut hits: Vec<Match> = Vec::new();
        for (pattern_idx, pattern) in patterns.iter_mut().enumerate() {
            if Some(pattern_idx) == base_pattern_idx || candidates.is_empty() {
                continue;
            }

            gathered.clear();
            gathered.extend(
                candidates
                    .iter()
                    .map(|m| haystacks[(m.index - haystack_index_offset) as usize].as_ref()),
            );
            hits.clear();
            Self::dispatch_pattern_into(pattern, &gathered, 0, &mut hits);

            // Backends emit matches in input order, so `hit.index` is the position of the
            // candidate it matched.
            if pattern.negated {
                // `retain` visits in order, so the counter tracks each candidate's position.
                let mut hits = hits.iter().peekable();
                let mut position = 0;
                candidates.retain(|_| {
                    let matched = hits.next_if(|hit| hit.index as usize == position).is_some();
                    position += 1;
                    !matched
                });
            } else {
                candidates = hits
                    .drain(..)
                    .map(|mut hit| {
                        let candidate = candidates[hit.index as usize];
                        hit.index = candidate.index;
                        hit.score = hit.score.saturating_add(candidate.score);
                        hit.exact |= candidate.exact;
                        #[cfg(feature = "match_end_col")]
                        {
                            hit.end_col = hit.end_col.max(candidate.end_col);
                        }
                        hit
                    })
                    .collect();
            }
        }

        matches.extend(candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CaseMatching, Config, Matching, Pattern, PatternConfig, SortStrategy};

    fn multi(query: &str, config: &Config) -> Matcher {
        Matcher::from_patterns(&Pattern::parse_query(query), config)
    }

    #[test]
    fn multi_pattern_negation() {
        let haystacks = ["foobar", "foo", "barfoo", "bar", "qux"];
        let config = Config::default().sort(SortStrategy::IndexAsc);
        let matches = multi("foo !bar", &config).match_list(&haystacks);
        assert_eq!(matches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn multi_pattern_negated_matching_modes() {
        let haystacks = ["foo/bar", "bar/foo", "foo", "foobar"];
        let config = Config::default().sort(SortStrategy::IndexAsc);

        // Prefix negation: excludes only haystacks starting with "bar"
        let matches = multi("foo !^bar", &config).match_list(&haystacks);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 2, 3]
        );

        // Suffix negation: excludes only haystacks ending with "bar"
        let matches = multi("foo !bar$", &config).match_list(&haystacks);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn multi_pattern_scores_sum() {
        let haystacks = ["foo", "xfoox", "bar"];
        let config = Config::default().sort(SortStrategy::IndexAsc);
        let single = Matcher::new("foo", &config).match_list(&haystacks);
        let combined = multi("foo foo", &config).match_list(&haystacks);

        assert_eq!(combined.len(), single.len());
        for (c, s) in combined.iter().zip(&single) {
            assert_eq!(c.index, s.index);
            assert_eq!(c.score, s.score * 2);
            assert_eq!(c.exact, s.exact);
            #[cfg(feature = "match_end_col")]
            assert_eq!(c.end_col, s.end_col);
        }
    }

    #[test]
    fn multi_pattern_all_negated() {
        let haystacks = ["foo", "bar", "xfoox", "qux"];
        let config = Config::default().sort(SortStrategy::IndexAsc);
        let matches = multi("!foo", &config).match_list(&haystacks);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert!(matches.iter().all(|m| m.score == 0));

        let matches = multi("!foo !qux", &config).match_list(&haystacks);
        assert_eq!(matches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn multi_pattern_contradiction_is_empty() {
        let haystacks = ["foo", "foobar"];
        let matches = multi("foo !foo", &Config::default()).match_list(&haystacks);
        assert!(matches.is_empty());
    }

    #[test]
    fn multi_pattern_score_sorted() {
        let haystacks = ["xfoobarx", "foobar", "zzz"];
        let matches = multi("foo bar", &Config::default()).match_list(&haystacks);
        assert_eq!(matches.len(), 2);
        assert!(matches.is_sorted());
        assert_eq!(matches[0].index, 1);
    }

    #[test]
    fn multi_pattern_match_iter_matches_match_list() {
        let haystacks = ["foobar", "foo", "barfoo", "bar", "qux", "FooBar"];
        for query in ["foo !bar", "foo bar", "!foo", "^foo bar$", "foo !^bar"] {
            let config = Config::default().sort(SortStrategy::IndexAsc);
            let mut matcher = multi(query, &config);
            let from_iter = matcher.match_iter(haystacks.iter()).collect::<Vec<_>>();
            let from_list = matcher.match_list(&haystacks);
            assert_eq!(from_iter, from_list, "query: {query:?}");
        }
    }

    #[test]
    fn multi_pattern_match_list_indices_matches_match_list() {
        let haystacks = ["foobar", "foo", "barfoo", "bar", "qux", "FooBar"];
        for query in ["foo !bar", "foo bar", "!foo", "foo fo"] {
            let config = Config::default().sort(SortStrategy::IndexAsc);
            let mut matcher = multi(query, &config);
            let matches = matcher.match_list(&haystacks);
            let indices = matcher.match_list_indices(&haystacks);

            assert_eq!(matches.len(), indices.len(), "query: {query:?}");
            for (m, i) in matches.iter().zip(&indices) {
                assert_eq!(m.index, i.index, "query: {query:?}");
                assert_eq!(m.score, i.score, "query: {query:?}");
                assert_eq!(m.exact, i.exact, "query: {query:?}");
                // Indices must be strictly descending (reverse order, deduped)
                assert!(
                    i.indices.windows(2).all(|w| w[0] > w[1]),
                    "query: {query:?}, indices: {:?}",
                    i.indices
                );
            }
        }
    }

    #[test]
    fn multi_pattern_overlapping_indices_deduped() {
        let mut matcher = multi("foo fo", &Config::default());
        let indices = matcher.match_list_indices(&["foo"]);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].indices, vec![2, 1, 0]);
    }

    #[test]
    fn pattern_matching_override_matches_config() {
        let haystacks = ["fooX", "xfoo", "foo"];
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let from_pattern = Matcher::from_patterns(
            &[Pattern::new(
                "foo",
                PatternConfig::default().matching(Some(Matching::Prefix)),
            )],
            &config,
        )
        .match_list(&haystacks);
        let from_config =
            Matcher::new("foo", &config.matching(Matching::Prefix)).match_list(&haystacks);
        assert_eq!(from_pattern, from_config);
    }

    #[test]
    fn set_config_preserves_pattern_matching_override() {
        let haystacks = ["fooX", "xfoo"];
        let config = Config::default().sort(SortStrategy::IndexAsc);
        let mut matcher = multi("^foo", &config);
        matcher.set_config(config.max_typos(None));

        let matches = matcher.match_list(&haystacks);
        assert_eq!(matches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn set_pattern_reverts_to_literal_matching() {
        let config = Config::default().sort(SortStrategy::IndexAsc);
        let mut matcher = multi("^foo", &config);
        assert_eq!(matcher.patterns(), &[Pattern::parse("^foo")]);
        assert_eq!(matcher.match_list(&["foobar", "^foo"]).len(), 1);

        // Same needle string, but the matcher must rebuild to match it literally
        matcher.set_pattern("^foo");
        let matches = matcher.match_list(&["foobar", "^foo"]);
        assert_eq!(matches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn set_patterns_skips_rebuild_when_unchanged() {
        let config = Config::default().sort(SortStrategy::IndexAsc);
        let mut matcher = Matcher::new("foo", &config);
        matcher.set_patterns(&["foo".into()]);
        matcher.set_pattern("foo");
        assert_eq!(matcher.match_list(&["foobar"]).len(), 1);
    }

    #[test]
    fn pattern_max_typos_override_beats_config() {
        // "helloz" has a 'z' absent from "hello", so it needs one typo to match. The
        // config forbids typos, but the pattern raises the budget to one.
        let haystacks = ["hello", "world"];
        let config = Config::default()
            .max_typos(Some(0))
            .sort(SortStrategy::IndexAsc);

        let strict = Matcher::from_patterns(&["helloz".into()], &config).match_list(&haystacks);
        assert!(strict.is_empty());

        let lenient = Matcher::from_patterns(
            &[Pattern::new(
                "helloz",
                PatternConfig::default().max_typos(Some(1)),
            )],
            &config,
        )
        .match_list(&haystacks);
        assert_eq!(lenient.iter().map(|m| m.index).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn pattern_max_typos_override_applies_per_pattern() {
        // "foo" inherits the config's zero-typo budget while "barz" (with its absent
        // 'z') raises its own budget to one. So "foo" must match exactly but
        // "bar" may be a typo off.
        let haystacks = ["foo bar", "fox bar"];
        let config = Config::default()
            .max_typos(Some(0))
            .sort(SortStrategy::IndexAsc);
        let patterns = [
            Pattern::new("foo", PatternConfig::default()),
            Pattern::new("barz", PatternConfig::default().max_typos(Some(1))),
        ];
        let matches = Matcher::from_patterns(&patterns, &config).match_list(&haystacks);
        // "fox bar" drops out: "foo" can't match "fox" within zero typos
        assert_eq!(matches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn set_config_preserves_pattern_max_typos_override() {
        let haystacks = ["helloz"];
        let patterns = [Pattern::new(
            "hellozz",
            PatternConfig::default().max_typos(Some(1)),
        )];
        let mut matcher = Matcher::from_patterns(&patterns, &Config::default());
        // Rebuild under a stricter config; the pattern override must still win
        matcher.set_config(
            Config::default()
                .max_typos(Some(0))
                .sort(SortStrategy::IndexAsc),
        );
        assert_eq!(matcher.match_list(&haystacks).len(), 1);
    }

    #[test]
    fn multi_pattern_smart_case_per_pattern() {
        let haystacks = ["Foo BAR", "foo bar"];
        let config = Config::default()
            .casing(CaseMatching::Smart)
            .sort(SortStrategy::IndexAsc);
        // "Foo" is case sensitive (contains uppercase), "bar" is not
        let matches = multi("Foo bar", &config).match_list(&haystacks);
        assert_eq!(matches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn multi_pattern_unicode_per_pattern() {
        let haystacks = ["다나 foo", "dana foo", "다나"];
        let config = Config::default().sort(SortStrategy::IndexAsc);
        let matches = multi("다나 foo", &config).match_list(&haystacks);
        assert_eq!(matches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn from_patterns_empty_patterns_match_everything() {
        let haystacks = ["foo", "bar"];
        let mut matcher = Matcher::from_patterns(&[], &Config::default());
        assert_eq!(matcher.match_list(&haystacks).len(), 2);

        let mut matcher = multi("! ^$", &Config::default());
        assert_eq!(matcher.match_list(&haystacks).len(), 2);
    }
}
