use std::collections::BTreeMap;

use neo_frizbee::k_merge::k_merge_matches_by_score_then_index_desc;
use neo_frizbee::{
    CaseMatching, Config, Match, MatchIndices, Matcher, Matching, Pattern, PatternConfig, Scoring,
    SortStrategy,
};

// Did you know you could do this?? News to me
#[path = "../src/smith_waterman/backend/tests/generator.rs"]
mod generator;
use generator::{ByteCursor, run_generated_inputs, test_bound};

#[derive(Debug, Clone)]
struct ApiCase {
    needle: String,
    haystacks: Vec<String>,
    config: Config,
}

impl ApiCase {
    fn from_bytes(input: &[u8]) -> Self {
        let mut cursor = ByteCursor::new(input);
        let needle_len = cursor.len(32, &[0, 1, 2, 7, 8, 15, 16, 31, 32]);
        let haystack_count = cursor.len(32, &[0, 1, 2, 7, 8, 15, 16, 31, 32]);
        let haystacks = (0..haystack_count)
            .map(|_| {
                let len = cursor.len(96, &[0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96]);
                cursor.string(len)
            })
            .collect();

        let max_typos = match cursor.next() % 5 {
            0 => None,
            1 => Some(0),
            2 => Some(1),
            3 => Some(2),
            _ => Some((cursor.next() as u16) % 8),
        };
        let casing = match cursor.next() % 3 {
            0 => CaseMatching::Ignore,
            1 => CaseMatching::Smart,
            _ => CaseMatching::Respect,
        };
        let matching = match cursor.next() % 5 {
            0 => Matching::Fuzzy,
            1 => Matching::Exact,
            2 => Matching::Prefix,
            3 => Matching::Suffix,
            _ => Matching::Substring,
        };
        let sort = if cursor.bool() {
            SortStrategy::ScoreThenIndexAsc
        } else {
            SortStrategy::IndexAsc
        };

        Self {
            needle: cursor.string(needle_len),
            haystacks,
            config: Config::default()
                .max_typos(max_typos)
                .casing(casing)
                .matching(matching)
                .sort(sort),
        }
    }
}

#[test]
fn generated_public_api_properties() {
    run_generated_inputs(1024, test_bound(4096, 384), |input| {
        let case = ApiCase::from_bytes(input);
        assert_public_api_case(&case);
    });
}

fn assert_public_api_case(case: &ApiCase) {
    let one_shot = Matcher::new(&case.needle, &case.config).match_list(&case.haystacks);
    let mut matcher = Matcher::new(&case.needle, &case.config);
    let reusable = matcher.match_list(&case.haystacks);
    assert_match_views_eq("Matcher::match_list", &reusable, &one_shot);

    let one_shot_indices =
        Matcher::new(&case.needle, &case.config).match_list_indices(&case.haystacks);
    let mut matcher = Matcher::new(&case.needle, &case.config);
    let reusable_indices = matcher.match_list_indices(&case.haystacks);
    assert_eq!(
        indices_views(&reusable_indices),
        indices_views(&one_shot_indices),
        "Matcher::match_list_indices mismatch for {case:?}"
    );

    let parallel_one =
        Matcher::new(&case.needle, &case.config).match_list_parallel(&case.haystacks, 1);
    assert_match_views_eq("parallel threads=1", &parallel_one, &one_shot);

    for &threads in PARALLEL_THREAD_COUNTS {
        let parallel =
            Matcher::new(&case.needle, &case.config).match_list_parallel(&case.haystacks, threads);
        if case.config.sort == SortStrategy::ScoreThenIndexAsc {
            assert_match_views_eq("parallel sorted", &parallel, &one_shot);
        } else {
            assert_eq!(
                sorted_match_views(&parallel),
                sorted_match_views(&one_shot),
                "parallel unsorted multiset mismatch for threads={threads} case={case:?}"
            );
        }
    }

    assert_indices_contract(case, &one_shot, &one_shot_indices);
}

fn assert_indices_contract(case: &ApiCase, matches: &[Match], indices: &[MatchIndices]) {
    let match_set = matches
        .iter()
        .map(|match_| (match_.index, (match_.score, match_.exact)))
        .collect::<BTreeMap<_, _>>();
    let indices_set = indices
        .iter()
        .map(|match_| (match_.index, (match_.score, match_.exact)))
        .collect::<BTreeMap<_, _>>();

    for match_ in indices {
        assert!(
            (match_.index as usize) < case.haystacks.len(),
            "index {} is out of bounds for {case:?}",
            match_.index
        );
        assert_eq!(
            match_set.get(&match_.index),
            Some(&(match_.score, match_.exact)),
            "indices result is not present in match_list for {case:?}"
        );

        let haystack = case.haystacks[match_.index as usize].as_bytes();
        assert!(
            match_
                .indices
                .windows(2)
                .all(|window| window[0] > window[1]),
            "indices are not reverse ordered for {case:?}: {:?}",
            match_.indices
        );
        assert!(
            match_.indices.len() <= case.needle.len(),
            "too many indices for {case:?}: {:?}",
            match_.indices
        );
        for &index in &match_.indices {
            assert!(
                index < haystack.len() as u32,
                "index {index} out of bounds for haystack len {} in {case:?}",
                haystack.len()
            );
        }
    }

    if case.config.max_typos.is_none() || case.config.matching != Matching::Fuzzy {
        assert_eq!(
            indices_set, match_set,
            "indices and matches should agree exactly without typo filtering for {case:?}"
        );
    }
}

#[cfg(not(feature = "match_end_col"))]
type MatchView = (u16, u32, bool);

#[cfg(feature = "match_end_col")]
type MatchView = (u16, u32, bool, u16);

#[cfg(not(feature = "match_end_col"))]
fn match_view(match_: &Match) -> MatchView {
    (match_.score, match_.index, match_.exact)
}

#[cfg(feature = "match_end_col")]
fn match_view(match_: &Match) -> MatchView {
    (match_.score, match_.index, match_.exact, match_.end_col)
}

fn match_views(matches: &[Match]) -> Vec<MatchView> {
    matches.iter().map(match_view).collect()
}

fn sorted_match_views(matches: &[Match]) -> Vec<MatchView> {
    let mut views = match_views(matches);
    views.sort();
    views
}

fn assert_match_views_eq(label: &str, got: &[Match], want: &[Match]) {
    assert_eq!(
        match_views(got),
        match_views(want),
        "{label} mismatch: got={got:?} want={want:?}"
    );
}

fn indices_views(matches: &[MatchIndices]) -> Vec<(u16, u32, bool, Vec<u32>)> {
    matches
        .iter()
        .map(|match_| {
            (
                match_.score,
                match_.index,
                match_.exact,
                match_.indices.clone(),
            )
        })
        .collect()
}

/// The haystack indices of a match list, in result order.
fn match_indices(matches: &[Match]) -> Vec<u32> {
    matches.iter().map(|match_| match_.index).collect()
}

fn mtch(score: u16, index: u32) -> Match {
    Match {
        score,
        index,
        exact: false,
        #[cfg(feature = "match_end_col")]
        end_col: 0,
    }
}

/// `len` `"nomatch-{i}"` haystacks with the given indices overwritten — used to
/// place matches at specific chunk boundaries in the parallel tests.
fn haystacks_with(len: usize, patches: &[(usize, &str)]) -> Vec<String> {
    let mut haystacks = (0..len)
        .map(|index| format!("nomatch-{index}"))
        .collect::<Vec<_>>();
    for &(index, value) in patches {
        haystacks[index] = value.to_string();
    }
    haystacks
}

const PARALLEL_CHUNK_SIZE: usize = if cfg!(miri) { 8 } else { 2048 };
const PARALLEL_THREAD_COUNTS: &[usize] = if cfg!(miri) { &[2] } else { &[2, 3, 8] };
const BOUNDARY_TEST_THREADS: usize = if cfg!(miri) { 2 } else { 8 };

#[derive(Debug, Clone)]
struct MultiPatternCase {
    patterns: Vec<Pattern>,
    haystacks: Vec<String>,
    config: Config,
}

impl MultiPatternCase {
    fn from_bytes(input: &[u8]) -> Self {
        let mut cursor = ByteCursor::new(input);
        let pattern_count = 1 + (cursor.next() as usize) % 3;
        let patterns = (0..pattern_count)
            .map(|_| {
                let matching = match cursor.next() % 6 {
                    0 => None,
                    1 => Some(Matching::Fuzzy),
                    2 => Some(Matching::Exact),
                    3 => Some(Matching::Prefix),
                    4 => Some(Matching::Suffix),
                    _ => Some(Matching::Substring),
                };
                let negated = cursor.bool();
                let needle_len = cursor.len(8, &[0, 1, 2, 3, 7, 8]);
                Pattern::new(
                    &cursor.string(needle_len),
                    PatternConfig::default().matching(matching),
                )
                .negated(negated)
            })
            .collect();

        let haystack_count = cursor.len(24, &[0, 1, 2, 7, 8, 15, 16, 24]);
        let haystacks = (0..haystack_count)
            .map(|_| {
                let len = cursor.len(48, &[0, 1, 2, 7, 8, 15, 16, 31, 32, 48]);
                cursor.string(len)
            })
            .collect();

        let max_typos = match cursor.next() % 4 {
            0 => None,
            1 => Some(0),
            2 => Some(1),
            _ => Some(2),
        };
        let casing = match cursor.next() % 3 {
            0 => CaseMatching::Ignore,
            1 => CaseMatching::Smart,
            _ => CaseMatching::Respect,
        };
        let matching = match cursor.next() % 5 {
            0 => Matching::Fuzzy,
            1 => Matching::Exact,
            2 => Matching::Prefix,
            3 => Matching::Suffix,
            _ => Matching::Substring,
        };

        Self {
            patterns,
            haystacks,
            config: Config::default()
                .max_typos(max_typos)
                .casing(casing)
                .matching(matching),
        }
    }
}

/// Reference implementation: match each pattern independently with single-atom
/// `match_list` calls, then intersect the non-negated patterns (summing scores)
/// and subtract the negated ones.
fn reference_multi_pattern(case: &MultiPatternCase) -> Vec<Match> {
    let config = case.config.clone().sort(SortStrategy::IndexAsc);
    let active = case
        .patterns
        .iter()
        .filter(|pattern| !pattern.needle.is_empty())
        .collect::<Vec<_>>();
    if active.is_empty() {
        return (0..case.haystacks.len()).map(Match::from_index).collect();
    }

    let per_pattern = active
        .iter()
        .map(|pattern| {
            let matching = pattern.config.matching.unwrap_or(case.config.matching);
            Matcher::new(&pattern.needle, &config.clone().matching(matching))
                .match_list(&case.haystacks)
                .into_iter()
                .map(|match_| (match_.index, match_))
                .collect::<BTreeMap<_, _>>()
        })
        .collect::<Vec<_>>();

    (0..case.haystacks.len() as u32)
        .filter_map(|index| {
            let mut combined = Match::from_index(index as usize);
            for (pattern, matches) in active.iter().zip(&per_pattern) {
                let hit = matches.get(&index);
                if pattern.negated {
                    if hit.is_some() {
                        return None;
                    }
                } else {
                    let hit = hit?;
                    combined.score = combined.score.saturating_add(hit.score);
                    combined.exact |= hit.exact;
                    #[cfg(feature = "match_end_col")]
                    {
                        combined.end_col = combined.end_col.max(hit.end_col);
                    }
                }
            }
            Some(combined)
        })
        .collect()
}

#[test]
fn generated_multi_pattern_properties() {
    run_generated_inputs(512, test_bound(4096, 256), |input| {
        let case = MultiPatternCase::from_bytes(input);
        assert_multi_pattern_case(&case);
    });
}

fn assert_multi_pattern_case(case: &MultiPatternCase) {
    let reference = reference_multi_pattern(case);

    let index_config = case.config.clone().sort(SortStrategy::IndexAsc);
    let mut matcher = Matcher::from_patterns(&case.patterns, &index_config);
    let matches = matcher.match_list(&case.haystacks);
    assert_match_views_eq("multi-pattern match_list", &matches, &reference);

    let by_index = reference
        .iter()
        .map(|match_| (match_.index, match_))
        .collect::<BTreeMap<_, _>>();
    for (index, haystack) in case.haystacks.iter().enumerate() {
        let one = matcher.match_one(haystack, index as u32);
        match by_index.get(&(index as u32)) {
            Some(&want) => {
                let got =
                    one.unwrap_or_else(|| panic!("match_one missing index {index} for {case:?}"));
                assert_eq!(
                    match_view(&got),
                    match_view(want),
                    "match_one mismatch at index {index} for {case:?}"
                );
            }
            None => assert!(
                one.is_none(),
                "match_one unexpectedly matched index {index} for {case:?}"
            ),
        }
    }

    let score_config = case.config.clone().sort(SortStrategy::ScoreThenIndexAsc);
    let mut matcher = Matcher::from_patterns(&case.patterns, &score_config);
    let sorted = matcher.match_list(&case.haystacks);
    assert!(sorted.is_sorted(), "unsorted result for {case:?}");
    assert_eq!(
        sorted_match_views(&sorted),
        sorted_match_views(&reference),
        "multi-pattern sorted multiset mismatch for {case:?}"
    );

    for &threads in PARALLEL_THREAD_COUNTS {
        let parallel = matcher.match_list_parallel(&case.haystacks, threads);
        assert_match_views_eq("multi-pattern parallel", &parallel, &sorted);
    }
}

#[test]
fn empty_needle_matches_everything() {
    let haystacks = ["foo", "bar"];
    let config = Config::default();

    let matches = Matcher::new("", &config).match_list(&haystacks);
    assert_eq!(
        match_views(&matches),
        match_views(&[Match::from_index(0), Match::from_index(1)])
    );

    let indices = Matcher::new("", &config).match_list_indices(&haystacks);
    assert_eq!(
        indices_views(&indices),
        indices_views(&[MatchIndices::from_index(0), MatchIndices::from_index(1)])
    );
}

#[test]
fn exact_match_flag_tracks_full_haystack_match() {
    let haystacks = ["deadbe", "deadbeef", "deadbe", "deadbf", "xxdeadbexx"];
    let matches = Matcher::new("deadbe", &Config::default()).match_list(&haystacks);

    let exact_by_index = matches
        .iter()
        .map(|match_| (match_.index, match_.exact))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(exact_by_index.get(&0), Some(&true));
    assert_eq!(exact_by_index.get(&1), Some(&false));
    assert_eq!(exact_by_index.get(&2), Some(&true));
    assert_eq!(exact_by_index.get(&4), Some(&false));
}

#[test]
fn unicode_matcher_zero_typo_uses_byte_offsets_and_exact_flags() {
    let haystacks = vec![
        "xxإنyy".to_string(),
        "إن".to_string(),
        "\u{06e5}\u{0606}".to_string(),
        "nomatch".to_string(),
        "x".repeat(65),
    ];
    let config = Config::default()
        .max_typos(Some(0))
        .sort(SortStrategy::IndexAsc);

    let matches = Matcher::new("إن", &config).match_list(&haystacks);
    let mut matcher = Matcher::new("إن", &config);
    let matcher_matches = matcher.match_list(&haystacks);
    assert_match_views_eq("unicode Matcher::match_list", &matcher_matches, &matches);
    assert_eq!(
        matches
            .iter()
            .map(|match_| (match_.index, match_.exact))
            .collect::<Vec<_>>(),
        vec![(0, false), (1, true)]
    );

    let indices = Matcher::new("إن", &config).match_list_indices(&haystacks);
    let mut matcher = Matcher::new("إن", &config);
    let matcher_indices = matcher.match_list_indices(&haystacks);
    assert_eq!(indices_views(&matcher_indices), indices_views(&indices));
    assert_eq!(indices.len(), 2);
    assert_eq!((indices[0].index, indices[0].exact), (0, false));
    assert_eq!(indices[0].indices, vec![5, 4, 3, 2]);
    assert_eq!((indices[1].index, indices[1].exact), (1, true));
    assert_eq!(indices[1].indices, vec![3, 2, 1, 0]);
}

#[test]
fn unicode_matcher_indices_cover_gaps_and_chunk_boundaries() {
    let config = Config::default()
        .max_typos(None)
        .sort(SortStrategy::IndexAsc);

    let gap_haystacks = ["é😀x"];
    let gap_matches = Matcher::new("éx", &config).match_list(&gap_haystacks);
    let gap_indices = Matcher::new("éx", &config).match_list_indices(&gap_haystacks);
    assert_eq!(gap_indices.len(), 1);
    assert_eq!(gap_indices[0].indices, vec![6, 1, 0]);
    assert_eq!(
        (gap_indices[0].score, gap_indices[0].exact),
        (gap_matches[0].score, gap_matches[0].exact)
    );

    let boundary_haystacks = ["_______😀x"];
    let boundary_matches = Matcher::new("😀x", &config).match_list(&boundary_haystacks);
    let boundary_indices = Matcher::new("😀x", &config).match_list_indices(&boundary_haystacks);
    assert_eq!(boundary_indices.len(), 1);
    assert_eq!(boundary_indices[0].indices, vec![11, 10, 9, 8, 7]);
    assert_eq!(
        (boundary_indices[0].score, boundary_indices[0].exact),
        (boundary_matches[0].score, boundary_matches[0].exact)
    );
}

#[test]
fn long_prefiltered_fallback_preserves_scores_and_original_byte_offsets() {
    let haystack = format!("xa{}b", "_".repeat(1200));
    let scoring = Scoring {
        gap_open_penalty: 0,
        gap_extend_penalty: 0,
        capitalization_bonus: 0,
        delimiter_bonus: 0,
        ..Scoring::default()
    };
    let filtered_config = Config::default()
        .scoring(scoring)
        .sort(SortStrategy::IndexAsc);
    let unfiltered_config = filtered_config.clone().max_typos(None);

    let filtered = Matcher::new("ab", &filtered_config).match_list(&[&haystack]);
    let unfiltered = Matcher::new("ab", &unfiltered_config).match_list(&[&haystack]);
    assert_match_views_eq("long prefiltered fallback", &filtered, &unfiltered);

    let filtered_indices = Matcher::new("ab", &filtered_config).match_list_indices(&[&haystack]);
    let unfiltered_indices =
        Matcher::new("ab", &unfiltered_config).match_list_indices(&[&haystack]);
    assert_eq!(
        indices_views(&filtered_indices),
        indices_views(&unfiltered_indices)
    );
    assert_eq!(filtered_indices[0].indices, vec![1202, 1]);

    let unicode_haystack = format!("xé{}b", "_".repeat(1200));
    let filtered = Matcher::new("éb", &filtered_config).match_list(&[&unicode_haystack]);
    let unfiltered = Matcher::new("éb", &unfiltered_config).match_list(&[&unicode_haystack]);
    assert_match_views_eq("long Unicode prefiltered fallback", &filtered, &unfiltered);

    let filtered_indices =
        Matcher::new("éb", &filtered_config).match_list_indices(&[&unicode_haystack]);
    let unfiltered_indices =
        Matcher::new("éb", &unfiltered_config).match_list_indices(&[&unicode_haystack]);
    assert_eq!(
        indices_views(&filtered_indices),
        indices_views(&unfiltered_indices)
    );
    assert_eq!(filtered_indices[0].indices, vec![1203, 2, 1]);
}

#[test]
fn unicode_matcher_typo_prefilter_counts_scalar_values() {
    let haystacks = ["ن", "😀", "x"];
    let config = Config::default()
        .max_typos(Some(1))
        .sort(SortStrategy::IndexAsc);

    let matches = Matcher::new("إن", &config).match_list(&haystacks);
    assert_eq!(match_indices(&matches), vec![0]);

    let many_typo_config = Config::default()
        .max_typos(Some(2))
        .sort(SortStrategy::IndexAsc);
    let matches = Matcher::new("éन😀", &many_typo_config).match_list(&haystacks);
    assert_eq!(match_indices(&matches), vec![1]);
}

#[test]
fn case_matching_modes_apply_to_matches_and_indices() {
    let haystacks = ["foo", "FOO", "fOo", "xxfooxx"];
    let config = Config::default().sort(SortStrategy::IndexAsc);
    assert_eq!(
        match_indices(&Matcher::new("foo", &config).match_list(&haystacks)),
        vec![0, 1, 2, 3]
    );

    let config = Config::default()
        .casing(CaseMatching::Respect)
        .sort(SortStrategy::IndexAsc);
    assert_eq!(
        match_indices(&Matcher::new("foo", &config).match_list(&haystacks)),
        vec![0, 3]
    );
    assert_eq!(
        Matcher::new("foo", &config)
            .match_list_indices(&haystacks)
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![0, 3]
    );

    let config = Config::default()
        .casing(CaseMatching::Smart)
        .sort(SortStrategy::IndexAsc);
    assert_eq!(
        match_indices(&Matcher::new("FoO", &config).match_list(&["foo", "FOO", "FoO", "xxFoOxx"]),),
        vec![2, 3]
    );
}

#[test]
fn long_needle_scores_saturate() {
    // Needles past `max_needle_len` match with scores capped at u16::MAX instead
    // of panicking
    let scoring = Scoring {
        match_score: u16::MAX / 4 + 1,
        ..Scoring::default()
    };
    let needle = "aaaa";
    assert!(needle.len() > scoring.max_needle_len());
    let matches = Matcher::new(needle, &Config::default().scoring(scoring)).match_list(&[needle]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].score, u16::MAX);
}

#[test]
fn zero_parallel_threads_uses_available_parallelism() {
    let haystacks = ["abc", "xabc", "zzz"];
    let mut matcher = Matcher::new("abc", &Config::default());
    let sequential = matcher.match_list(&haystacks);
    assert_eq!(matcher.match_list_parallel(&haystacks, 0), sequential);
}

#[test]
fn parallel_chunk_boundaries_match_sequential() {
    let chunk_size = PARALLEL_CHUNK_SIZE;
    let haystacks = haystacks_with(
        2 * chunk_size + 5,
        &[
            (0, "abc"),
            (chunk_size - 1, "xabc"),
            (chunk_size, "abxc"),
            (chunk_size + 1, "alpha/beta/abc"),
            (2 * chunk_size - 1, "ABC"),
            (2 * chunk_size, "a_b_c"),
            (2 * chunk_size + 4, "zabc"),
        ],
    );

    let sorted = Config::default();
    let sequential = Matcher::new("abc", &sorted).match_list(&haystacks);
    let parallel_one = Matcher::new("abc", &sorted).match_list_parallel(&haystacks, 1);
    assert_match_views_eq("single-thread chunk boundary", &parallel_one, &sequential);

    let parallel =
        Matcher::new("abc", &sorted).match_list_parallel(&haystacks, BOUNDARY_TEST_THREADS);
    assert_match_views_eq("sorted chunk boundary", &parallel, &sequential);

    let unsorted = Config::default().sort(SortStrategy::IndexAsc);
    let sequential = Matcher::new("abc", &unsorted).match_list(&haystacks);
    let parallel =
        Matcher::new("abc", &unsorted).match_list_parallel(&haystacks, BOUNDARY_TEST_THREADS);
    assert_eq!(
        sorted_match_views(&parallel),
        sorted_match_views(&sequential)
    );
}

#[test]
fn sorted_parallel_equal_scores_use_index_tiebreaking_across_chunks() {
    let chunk_size = PARALLEL_CHUNK_SIZE;
    let haystacks = haystacks_with(
        2 * chunk_size + 1,
        &[
            (chunk_size - 1, "abc"),
            (chunk_size, "abc"),
            (2 * chunk_size, "abc"),
        ],
    );

    let config = Config::default();
    let sequential = Matcher::new("abc", &config).match_list(&haystacks);
    assert_eq!(
        match_indices(&sequential),
        vec![
            (chunk_size - 1) as u32,
            chunk_size as u32,
            (2 * chunk_size) as u32
        ]
    );

    let parallel =
        Matcher::new("abc", &config).match_list_parallel(&haystacks, BOUNDARY_TEST_THREADS);
    assert_match_views_eq("equal-score chunk tie", &parallel, &sequential);
}

#[test]
fn merges_by_score_then_index_desc() {
    let runs = vec![
        vec![mtch(100, 3), mtch(80, 5), mtch(20, 1)],
        vec![mtch(100, 2), mtch(90, 4), mtch(80, 0)],
    ];

    assert_eq!(
        k_merge_matches_by_score_then_index_desc(runs),
        vec![
            mtch(100, 3),
            mtch(100, 2),
            mtch(90, 4),
            mtch(80, 5),
            mtch(80, 0),
            mtch(20, 1),
        ]
    );
}

#[test]
fn score_prefer_higher_index_matches_sequential_and_parallel() {
    let chunk_size = PARALLEL_CHUNK_SIZE;
    let haystacks = haystacks_with(
        2 * chunk_size + 5,
        &[
            (0, "abc"),
            (1, "xabc"),
            (chunk_size - 1, "abc"),
            (chunk_size, "a_b_c"),
            (2 * chunk_size, "abc"),
            (2 * chunk_size + 4, "zabc"),
        ],
    );
    let config = Config::default().sort(SortStrategy::ScoreThenIndexDesc);
    let mut matcher = Matcher::new("abc", &config);

    let sequential = matcher.match_list(&haystacks);
    assert!(sequential.windows(2).all(|pair| {
        pair[0].score > pair[1].score
            || (pair[0].score == pair[1].score && pair[0].index > pair[1].index)
    }));

    let parallel = matcher.match_list_parallel(&haystacks, BOUNDARY_TEST_THREADS);
    assert_match_views_eq("prefer higher index", &parallel, &sequential);
}

#[test]
fn reverse_index_matches_sequential_and_parallel() {
    let chunk_size = PARALLEL_CHUNK_SIZE;
    let haystacks = haystacks_with(
        2 * chunk_size + 5,
        &[
            (0, "abc"),
            (1, "xabc"),
            (chunk_size - 1, "abc"),
            (chunk_size, "a_b_c"),
            (2 * chunk_size, "abc"),
            (2 * chunk_size + 4, "zabc"),
        ],
    );
    let config = Config::default().sort(SortStrategy::IndexDesc);
    let mut matcher = Matcher::new("abc", &config);

    let sequential = matcher.match_list(&haystacks);
    assert!(
        sequential
            .windows(2)
            .all(|pair| pair[0].index > pair[1].index)
    );

    let parallel = matcher.match_list_parallel(&haystacks, BOUNDARY_TEST_THREADS);
    assert_match_views_eq("reverse index", &parallel, &sequential);
}

#[test]
fn public_indices_are_reverse_byte_offsets() {
    let haystacks = ["xabcx", "a_b_c", "nomatch"];
    let config = Config::default().sort(SortStrategy::IndexAsc);

    let matches = Matcher::new("abc", &config).match_list_indices(&haystacks);
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].index, 0);
    assert_eq!(matches[0].indices, vec![3, 2, 1]);
    assert_eq!(matches[1].index, 1);
    assert_eq!(matches[1].indices, vec![4, 2, 0]);
}

#[test]
#[cfg(feature = "match_end_col")]
fn match_end_col_survives_prefilter_offsets() {
    let config = Config::default()
        .max_typos(None)
        .sort(SortStrategy::IndexAsc);
    let matches = Matcher::new("abc", &config).match_list(&["xabcx", "abcdef", "xxabc"]);
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].end_col, 3);
    assert_eq!(matches[1].end_col, 2);
    assert_eq!(matches[2].end_col, 4);
}

#[test]
fn custom_scoring_stays_within_overflow_guard() {
    let config = Config::default().scoring(Scoring {
        match_score: 8,
        matching_case_bonus: 1,
        ..Scoring::default()
    });
    let matches = Matcher::new("abc", &config).match_list(&["abc", "a_b_c"]);
    assert_eq!(matches.len(), 2);
}
