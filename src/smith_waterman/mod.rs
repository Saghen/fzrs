//! The [Smith Waterman algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) performs local sequence alignment ([explanation](https://kaell.se/bibook/pairwise/waterman.html)), originally designed to find similar sequences between two DNA strings. Guaranteed to find the optimal alignment and supports typos.
//!
//! The algorithm's time and space complexity of O(nm) led to plenty of research on parallelization. Each cell in the matrix has a data dependency on the cell to the left, up, and left-up diagonal. For biology, DNA sequences are typically quite large (m > 1000), so most of the parallelization approaches focused on large matrices ([see this paper for common parallelization techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822)).
//!
//! As a fuzzy matcher, the matrices in Frizbee are typically much smaller than those in DNA alignment (m < 128). Frizbee uses an approach similar to [sequential layout](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822/#Sec11), except the horizontal (vertical in the paper, but flipped in frizbee) data dependency is applied immediately. This approach supports [affine gaps](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Affine).
//!
//! ```text
//! needle: "foo"
//! haystack: "some/long/foo/path"
//!
//! // assuming 4 lane SIMD for simplicity
//! // in reality, we use anywhere from 8-64 SIMD lanes
//!
//! score_matrix:
//!    [s   o   m   e]   [/   l   o   n]   [g   /   f   o]   [o   /   p   a]   [t   h   _   _]
//! f  [0   0   0   0]   [0   0   0   0]   [0   0   16  11]  [10  9   8   7]   [6   5   4   3]
//! o  [0   16  11  10]  [9   8   16  11]  [10  9   8   32]  [27  26  25  24]  [23  22  21  20]
//! o  [0   16  11  10]  [9   8   24  19]  [18  17  16  24]  [48  43  42  41]  [40  39  38  37]
//!
//! // for the SIMD register at row 2, col 1, we would start with
//!
//! needle:      [o   o   o   o]
//! haystack:    [/   l   o   n]
//! match mask:  [N   N   Y   N]
//!
//! diagonal:    [10  9   8   16]
//! up:          [9   8   16  11]
//! current:     [8   7   24  9]
//!
//! // now we propagate the left data dependency
//!
//! left:        [0   16  11  10]
//! // shift current right by 1 element, filling in with right most element from left
//! shifted:     [10  8   7   24]
//! // decay by gap extend penalty (1)
//! // last element decayed by 5 (gap open penalty) instead of 1 (gap extend penalty)
//! // because the previous element matched (affine gaps)
//! decayed:     [9   7   6   19]
//! // max with current
//! current:     [9   7   24  19]
//! // repeat for shifting by 2 elements
//! shifted:     [11  10  9   7]
//! decayed:     [9   8   7   5] // gap extend penalty * 2 or gap open penalty + extend penalty
//! current:     [9   8   24  19]
//!
//! final:       [8   7   24  19]
//! ```
//!
//! Frizbee previously used inter-sequence parallelism (one needle, $LANES
//! haystacks) but this performed about the same as sequential layout due to
//! requiring interleaving the haystacks and bucketing based on haystack length,
//! while performing worse in parallel due to the required bucketing.

use crate::{Scoring, prefilter::UnicodeChar};
use alloc::{string::String, vec::Vec};
use backend::Backend;
#[cfg(target_arch = "x86_64")]
use backend::{BackendAVX, BackendAVX512, BackendAVX512U8, BackendAVXU8, BackendSSE, BackendSSEU8};
#[cfg(target_arch = "aarch64")]
use backend::{BackendNEON, BackendNEONU8};
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
use backend::{BackendScalar8, BackendScalar16U8};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use backend::{BackendWasm, BackendWasmU8};
use matrix::Matrix;

mod algo;
mod alignment;
mod alignment_iter;
pub(crate) mod backend;
mod greedy;
mod matrix;

use alignment_iter::AlignmentPathIter;

#[cfg(target_arch = "x86_64")]
pub type SmithWatermanAVX512U8 = SmithWaterman<BackendAVX512U8>;
#[cfg(target_arch = "x86_64")]
pub type SmithWatermanAVX512 = SmithWaterman<BackendAVX512>;
#[cfg(target_arch = "x86_64")]
pub type SmithWatermanSSE = SmithWaterman<BackendSSE>;
#[cfg(target_arch = "x86_64")]
pub type SmithWatermanSSEU8 = SmithWaterman<BackendSSEU8>;
#[cfg(target_arch = "x86_64")]
pub type SmithWatermanAVX = SmithWaterman<BackendAVX>;
#[cfg(target_arch = "x86_64")]
pub type SmithWatermanAVXU8 = SmithWaterman<BackendAVXU8>;
#[cfg(target_arch = "aarch64")]
pub type SmithWatermanNEON = SmithWaterman<BackendNEON>;
#[cfg(target_arch = "aarch64")]
pub type SmithWatermanNEONU8 = SmithWaterman<BackendNEONU8>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type SmithWatermanWasm = SmithWaterman<BackendWasm>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type SmithWatermanWasmU8 = SmithWaterman<BackendWasmU8>;
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
pub type SmithWatermanScalar = SmithWaterman<BackendScalar8>;
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
pub type SmithWatermanScalarU8 = SmithWaterman<BackendScalar16U8>;

/// Returns true if every possible Smith-Waterman matrix cell value for this
/// needle length and scoring config fits in a u8. The u8 backends are otherwise
/// identical to the u16 backends but with double the lane count (64 cells/chunk
/// on AVX-512, 32 on AVX2, 16 on SSE/NEON).
#[inline]
pub(crate) fn score_fits_in_u8(needle_len: usize, scoring: &Scoring) -> bool {
    let max_constant = (scoring.match_score as usize + scoring.mismatch_penalty as usize)
        .max(scoring.gap_open_penalty as usize)
        .max(scoring.gap_extend_penalty as usize)
        .max(scoring.matching_case_bonus as usize)
        .max(scoring.capitalization_bonus as usize)
        .max(scoring.delimiter_bonus as usize)
        .max(scoring.prefix_bonus as usize);
    if max_constant > u8::MAX as usize {
        return false;
    }

    // Gap propagation multiplies the gap extend penalty by up to 64x (AVX-512)
    let max_gap_penalty =
        64 * scoring.gap_extend_penalty as usize + scoring.gap_open_penalty as usize;
    if max_gap_penalty > u8::MAX as usize {
        return false;
    }

    let max_per_char = scoring.match_score as usize + scoring.max_per_char_bonus() as usize;
    let max_matrix_score = max_per_char * needle_len
        + scoring.max_one_time_bonus() as usize
        + scoring.prefix_bonus as usize;
    max_matrix_score + scoring.mismatch_penalty as usize <= u8::MAX as usize
}

#[derive(Debug, Clone)]
pub(crate) struct SmithWaterman<B: Backend> {
    needle: String,
    needle_simd: Vec<(B::Bytes, B::Bytes)>,
    needle_unicode: Vec<UnicodeChar>,
    case_sensitive: bool,
    scoring: Scoring,
    score_matrix: Matrix<B>,
    match_masks: Matrix<B>,
    unicode_pending_gap_open_masks: Vec<B::Score>,
    /// Number of LANES-wide chunks (incl. the leading zero column) actually
    /// consumed by the most recent `score_haystack` call. The matrix stride is
    /// always sized for `MAX_HAYSTACK_LEN` for zero-free reuse.
    haystack_chunks: usize,
}

pub(crate) trait Kernel: Clone + core::fmt::Debug + 'static {
    fn new(needle: &str, scoring: &Scoring, case_sensitive: bool) -> Self;
    fn is_available() -> bool;
    fn score_haystack_indices(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
        max_typos: Option<u16>,
    ) -> (u16, Vec<u32>);
    fn score_haystack_unicode_indices(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
        max_typos: Option<u16>,
    ) -> (u16, Vec<u32>);
    fn score_haystack(&mut self, haystack: &[u8], haystack_start_pos: usize) -> u16;
    fn score_haystack_unicode(&mut self, haystack: &[u8], haystack_start_pos: usize) -> u16;
    #[cfg(feature = "match_end_col")]
    fn match_end_col(&self, haystack: &[u8], unicode: bool) -> u16;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#const::*;
    use crate::smith_waterman::backend::BackendScalar8;

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWaterman::<BackendScalar8>::new(needle, &Scoring::default(), false);
        matcher.score_haystack(haystack.as_bytes(), true)
    }

    fn get_unicode_score(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWaterman::<BackendScalar8>::new(needle, &Scoring::default(), false);
        matcher.score_haystack_unicode(haystack.as_bytes(), true)
    }

    fn get_score_typos(needle: &str, haystack: &str, max_typos: u16) -> Option<u16> {
        get_score_typos_case(needle, haystack, max_typos, false)
    }

    fn get_score_typos_case(
        needle: &str,
        haystack: &str,
        max_typos: u16,
        case_sensitive: bool,
    ) -> Option<u16> {
        let mut matcher =
            SmithWaterman::<BackendScalar8>::new(needle, &Scoring::default(), case_sensitive);

        let score = matcher.score_haystack(haystack.as_bytes(), true);
        matcher
            .has_alignment_path(score, max_typos)
            .then_some(score)
    }

    fn get_indices(needle: &str, haystack: &str) -> Vec<u32> {
        let mut matcher = SmithWaterman::<BackendScalar8>::new(needle, &Scoring::default(), false);

        matcher
            .score_haystack_indices(haystack.as_bytes(), 0, None)
            .1
    }

    fn get_unicode_indices(needle: &str, haystack: &str) -> Vec<u32> {
        let mut matcher = SmithWaterman::<BackendScalar8>::new(needle, &Scoring::default(), false);

        matcher
            .score_haystack_unicode_indices(haystack.as_bytes(), 0, None)
            .1
    }

    #[test]
    fn test_score_basic() {
        assert_eq!(get_score("b", "abc"), CHAR_SCORE);
        assert_eq!(get_score("c", "abc"), CHAR_SCORE);
    }

    #[test]
    fn test_score_prefix() {
        assert_eq!(get_score("a", "abc"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("a", "aabc"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("a", "babc"), CHAR_SCORE);
    }

    #[test]
    fn test_score_exact_match() {
        assert_eq!(get_score("a", "a"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("abc", "abc"), 3 * CHAR_SCORE + PREFIX_BONUS);
    }

    #[test]
    fn unicode_score_counts_multibyte_scalars_once() {
        assert_eq!(get_unicode_score("é", "é"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_unicode_score("😀", "😀"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_unicode_score("éx", "éx"), 2 * CHAR_SCORE + PREFIX_BONUS);
    }

    #[test]
    fn unicode_gap_propagation_counts_skipped_scalars_once() {
        assert_eq!(
            get_unicode_score("éx", "ébx"),
            get_unicode_score("éx", "é😀x")
        );
        assert_eq!(
            get_unicode_score("ab", "aéb"),
            2 * CHAR_SCORE + PREFIX_BONUS - GAP_OPEN_PENALTY
        );
    }

    #[test]
    fn unicode_gap_propagation_handles_adjacent_scalar_end_then_body() {
        assert_eq!(
            get_unicode_score("ab", "aé😀b"),
            2 * CHAR_SCORE + PREFIX_BONUS - GAP_OPEN_PENALTY - GAP_EXTEND_PENALTY
        );
    }

    #[test]
    fn test_score_delimiter() {
        assert_eq!(get_score("-", "a--bc"), CHAR_SCORE);
        assert_eq!(get_score("b", "a-b"), CHAR_SCORE + DELIMITER_BONUS);
        assert_eq!(get_score("a", "a-b-c"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("b", "a--b"), CHAR_SCORE + DELIMITER_BONUS);
        assert_eq!(get_score("c", "a--bc"), CHAR_SCORE);
        assert_eq!(get_score("a", "-a--bc"), CHAR_SCORE + DELIMITER_BONUS);
    }

    #[test]
    fn test_score_no_delimiter_for_delimiter_chars() {
        assert_eq!(get_score("-", "a-bc"), CHAR_SCORE);
        assert_eq!(get_score("-", "a--bc"), CHAR_SCORE);
        assert!(get_score("a_b", "a_bb") > get_score("a_b", "a__b"));
    }

    #[test]
    fn test_score_affine_gap() {
        assert_eq!(
            get_score("test", "Uteost"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY
        );
        assert_eq!(
            get_score("test", "Uteoost"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY - GAP_EXTEND_PENALTY
        );
        assert_eq!(
            get_score("test", "Utooooeoooosoooot"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY * 3 - GAP_EXTEND_PENALTY * 9
        );
        assert_eq!(
            get_score("test", "Utooooooeoooooosoooooot"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY * 3 - GAP_EXTEND_PENALTY * 15
        );
    }

    #[test]
    fn test_score_capital_bonus() {
        assert_eq!(get_score("a", "A"), MATCH_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("A", "Aa"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("D", "forDist"), CHAR_SCORE + CAPITALIZATION_BONUS);
        assert_eq!(get_score("D", "foRDist"), CHAR_SCORE);
        assert_eq!(get_score("D", "FOR_DIST"), CHAR_SCORE + DELIMITER_BONUS);
    }

    #[test]
    fn test_score_prefix_beats_delimiter() {
        assert!(get_score("swap", "swap(test)") > get_score("swap", "iter_swap(test)"));
        assert!(get_score("_", "_private_member") > get_score("_", "public_member"));
    }

    #[test]
    fn test_score_prefix_beats_capitalization() {
        assert!(get_score("H", "HELLO") > get_score("H", "fooHello"));
    }

    #[test]
    fn test_score_continuous_beats_delimiter() {
        assert!(get_score("foo", "fooo") > get_score("foo", "f_o_o_o"));
    }

    #[test]
    fn test_score_continuous_beats_capitalization() {
        assert!(get_score("fo", "foo") > get_score("fo", "faOo"));
    }

    #[test]
    fn tie_prone_alignment_indices_are_stable() {
        assert_eq!(get_indices("aa", "aaa"), vec![1, 0]);
        assert_eq!(get_indices("ab", "abab"), vec![1, 0]);
        assert_eq!(get_indices("abc", "xabcabc"), vec![3, 2, 1]);
    }

    #[test]
    fn typo_threshold_distinguishes_mismatch_deletion_and_haystack_gap() {
        assert_eq!(get_score_typos("abc", "axc", 0), None);
        assert!(get_score_typos("abc", "axc", 1).is_some());

        assert_eq!(get_score_typos("abc", "ac", 0), None);
        assert!(get_score_typos("abc", "ac", 1).is_some());

        assert!(get_score_typos("abc", "abbc", 0).is_some());
    }

    #[test]
    fn one_long_gap_beats_repeated_gap_opens() {
        assert!(get_score("abc", "a111bc") > get_score("abc", "a1b1c"));
    }

    #[test]
    fn bonus_precedence_manual_cases() {
        assert!(get_score("b", "b") > get_score("b", "a-b"));
        assert!(get_score("b", "a-b") > get_score("b", "ab"));
        assert!(get_score("B", "aB") > get_score("b", "aB"));
    }

    #[test]
    fn case_sensitive_scoring_rejects_folded_bytes() {
        assert_eq!(
            get_score_typos_case("A", "A", 0, true),
            Some(CHAR_SCORE + PREFIX_BONUS)
        );
        assert_eq!(get_score_typos_case("A", "a", 0, true), None);
        assert_eq!(
            get_score_typos_case("A", "a", 0, false),
            Some(MATCH_SCORE + PREFIX_BONUS)
        );
    }

    #[cfg(feature = "match_end_col")]
    fn get_end_col(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWaterman::<BackendScalar8>::new(needle, &Scoring::default(), false);
        matcher.score_haystack(haystack.as_bytes(), true);
        matcher.match_end_col(haystack.as_bytes(), false)
    }

    #[cfg(feature = "match_end_col")]
    fn get_end_col_unicode(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWaterman::<BackendScalar8>::new(needle, &Scoring::default(), false);
        matcher.score_haystack_unicode(haystack.as_bytes(), true);
        matcher.match_end_col(haystack.as_bytes(), true)
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn test_end_col_unicode() {
        // The match column is the scalar start of "é" (byte 2), not the unwritten
        // byte-length row that previously returned 0
        assert_eq!(get_end_col_unicode("é", "xxé"), 2);
        assert_eq!(get_end_col_unicode("abc", "abcdef"), 2);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn test_end_col_basic() {
        assert_eq!(get_end_col("abc", "abcdef"), 2);
        assert_eq!(get_end_col("a", "abc"), 0);
        assert_eq!(get_end_col("c", "abc"), 2);
        assert_eq!(get_end_col("def", "abcdef"), 5);
        assert_eq!(get_end_col("def", "________________abcdef"), 21);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn long_input_end_col_uses_original_haystack_offsets() {
        let haystack = format!("{}abc", "x".repeat(510));
        assert_eq!(get_end_col("abc", &haystack), 512);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn huge_haystack_end_col_saturates() {
        let haystack = format!("a{}bc", "x".repeat(70000));
        assert_eq!(get_end_col("abc", &haystack), u16::MAX);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn long_input_boundary_end_cols_cover_matrix_and_greedy() {
        for (prefix_len, want) in [(509usize, 511u16), (510, 512)] {
            let haystack = format!("{}abc", "x".repeat(prefix_len));
            assert_eq!(get_end_col("abc", &haystack), want);
        }
    }

    #[test]
    fn test_score_typos() {
        assert_eq!(get_score_typos("foo", "Ufooo", 0), Some(CHAR_SCORE * 3));
        assert_eq!(get_score_typos("foo", "Ufo", 0), None);
        assert_eq!(
            get_score_typos("foo", "Ufo", 1),
            Some(CHAR_SCORE * 2 - GAP_OPEN_PENALTY)
        );
        assert_eq!(
            get_score_typos("foo", "Ufo", 2),
            Some(CHAR_SCORE * 2 - GAP_OPEN_PENALTY)
        );
        assert_eq!(get_score_typos("foo", "Uf", 1), None);
        assert_eq!(
            get_score_typos("foo", "Uf", 2),
            Some(CHAR_SCORE - GAP_OPEN_PENALTY - GAP_EXTEND_PENALTY)
        );
        assert_eq!(get_score_typos("foo", "U", 2), None);
        assert_eq!(get_score_typos("foo", "U", 3), Some(0));
        assert_eq!(get_score_typos("foo", "U", 4), Some(0));
    }

    #[test]
    fn test_indices_basic() {
        assert_eq!(get_indices("_", "abc"), vec![]);
        assert_eq!(get_indices("a", "abc"), vec![0]);
        assert_eq!(get_indices("b", "abc"), vec![1]);
        assert_eq!(get_indices("c", "abc"), vec![2]);
        assert_eq!(get_indices("ac", "________________abc"), vec![18, 16]);
        assert_eq!(get_indices("foo", "Uf"), vec![1]);
    }

    #[test]
    fn unicode_indices_expand_multibyte_scalars() {
        assert_eq!(get_unicode_indices("é", "é"), vec![1, 0]);
        assert_eq!(get_unicode_indices("😀", "😀"), vec![3, 2, 1, 0]);
        assert_eq!(get_unicode_indices("aé", "aé"), vec![2, 1, 0]);
    }

    #[test]
    fn unicode_indices_use_original_byte_offsets() {
        let mut matcher = SmithWaterman::<BackendScalar8>::new("é", &Scoring::default(), false);

        assert_eq!(
            matcher
                .score_haystack_unicode_indices("é".as_bytes(), 3, None)
                .1,
            vec![4, 3]
        );
    }

    #[test]
    fn unicode_indices_with_offset_trace_through_multibyte_haystack_gaps() {
        let mut matcher = SmithWaterman::<BackendScalar8>::new("éx", &Scoring::default(), false);

        assert_eq!(
            matcher
                .score_haystack_unicode_indices("é😀x".as_bytes(), 3, None)
                .1,
            vec![9, 4, 3]
        );
    }

    #[test]
    fn unicode_indices_trace_through_multibyte_haystack_gaps() {
        assert_eq!(get_unicode_indices("ab", "aéb"), vec![3, 0]);
        assert_eq!(get_unicode_indices("ab", "aé😀b"), vec![7, 0]);
        assert_eq!(get_unicode_indices("éx", "é😀x"), vec![6, 1, 0]);
    }

    #[test]
    fn unicode_indices_handle_repeated_scalars_and_chunk_boundaries() {
        assert_eq!(get_unicode_indices("éé", "ééé"), vec![3, 2, 1, 0]);
        assert_eq!(
            get_unicode_indices("😀x", "_______😀x"),
            vec![11, 10, 9, 8, 7]
        );
    }

    #[test]
    fn unicode_indices_do_not_split_multibyte_scalars_in_traceback() {
        // ensures that when we do traceback, we match all indices of multi-byte
        // unicode chars when they match
        assert_eq!(get_unicode_indices("😀.a", "..😀a"), vec![6, 1]);
        assert_eq!(get_unicode_indices("😀.é", "..😀é"), vec![7, 6, 1]);
        assert_eq!(get_unicode_indices("😀 a", "  😀a"), vec![6, 1]);
        assert_eq!(get_unicode_indices("😀é", "..😀é"), vec![7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn long_input_boundary_indices_stay_reverse_ordered() {
        for len in [1023, 1024, 1025] {
            let haystack = format!("{}abc", "x".repeat((len - 3) as usize));
            assert_eq!(get_score("abc", &haystack), 3 * CHAR_SCORE, "len={len}");
            assert_eq!(
                get_indices("abc", &haystack),
                vec![len - 1, len - 2, len - 3],
                "len={len}"
            );
        }
    }

    #[test]
    fn score_fits_in_u8_bounds_gap_extend_doubling() {
        assert!(score_fits_in_u8(4, &Scoring::default()));
        // The gap propagators scale the gap extend penalty by up to 64x on the widest
        // u8 backend, so 8 * 64 = 512 would wrap the u8 gap penalty
        let scoring = Scoring {
            gap_extend_penalty: 8,
            ..Scoring::default()
        };
        assert!(!score_fits_in_u8(4, &scoring));
    }
}
