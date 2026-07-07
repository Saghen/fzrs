//! Frizbee is a SIMD typo-resistant fuzzy string matcher written in Rust, with bindings for C, C++, Python and WASM. The core of the algorithm uses Smith-Waterman with affine gaps, similar to FZF. In the included benchmark, with typo resistance disabled, it outperforms [Nucleo](https://github.com/helix-editor/nucleo) by ~4x and [FZF](https://github.com/junegunn/fzf) by ~5x and supports multithreading, see [benchmarks](https://github.com/saghen/frizbee/blob/main/BENCHMARKS.md). When matching against unicode, it outperforms Nucleo and FZF by 20x.
//!
//! Used by [blink.cmp](https://github.com/saghen/blink.cmp), [atuin](https://github.com/atuinsh/atuin), [television](https://github.com/alexpasmantier/television), [skim](https://github.com/skim-rs/skim), and [fff](https://github.com/dmtrKovalenko/fff). Special thank you to [stefanboca](https://github.com/stefanboca) and [ii14](https://github.com/ii14)!
//!
//! For commercial support, please [contact me](mailto:frizbee@liam.super.fish). I'd be happy to work with you directly! Also, please consider [sponsoring me](https://github.com/sponsors/saghen).
//!
//! The core of the algorithm is Smith-Waterman with affine gaps and row-wise parallelism via SIMD. Besides the parallelism, this is the basis of other popular fuzzy matching algorithms like [FZF](https://github.com/junegunn/fzf) and [Nucleo](https://github.com/helix-editor/nucleo). The main properties of Smith-Waterman are:
//! - Always finds the best alignment
//! - Supports insertion (unmatched char in haystack, basis of fuzzy matching)
//! - Supports deletion (unmatched char in needle, basis of typo-resistance)
//! - Supports substitution (haystack and needle char mismatch, basis of
//!   typo-resistance)
//!
//! # Example: using `Matcher`
//!
//! `Matcher` compiles the pattern once, allocates memory for the Smith Waterman
//! matrix, and reuses the selected SIMD backend.
//!
//! Ideally, only construct these at most once per list. They're cheap to
//! construct, but end up being expensive if you construct them for each item in
//! your list.
//!
//! ```rust
//! use neo_frizbee::{Config, Matcher};
//!
//! let needle = "fBr";
//! let haystacks = ["fooBar", "foo_bar", "barfoo", "prelude", "println!"];
//!
//! let mut matcher = Matcher::new(needle, &Config::default());
//! let matches = matcher.match_list(&haystacks);
//! // or in parallel (8 threads, set to 0 to auto-detect)
//! let matches = matcher.match_list_parallel(&haystacks, 8);
//! ```
//!
//! # Example: using multi-pattern queries
//!
//! `Matcher::from_query` parses whitespace-separated atoms. Atom syntax can
//! control the matching mode:
//!
//! ```text
//! fuzzy  substring  prefix    suffix    exact    negated (combines with others)
//! foo    'foo       ^foo      foo$      ^foo$    !foo
//! ```
//!
//! ```rust
//! use neo_frizbee::{Config, Matcher};
//!
//! let haystacks = ["foo", "barfoo", "foobar", "bar/foo"];
//! let mut matcher = Matcher::from_query("foo !^bar", &Config::default());
//! let matches = matcher.match_list(&haystacks);
//! ```
//!
//! `Pattern::parse_query` returns the parsed patterns, so per-pattern config
//! can be applied before building the matcher. For example, setting the max
//! typos based on needle length:
//!
//! ```rust
//! use neo_frizbee::{Config, Matcher, Pattern};
//!
//! let haystacks = ["foo", "barfoo", "foobar", "bar/foo"];
//! let patterns = Pattern::parse_query("foo !^bar")
//!     .into_iter()
//!     .map(|pattern| {
//!         let max_typos = (pattern.needle.len() / 4) as u16;
//!         pattern.max_typos(Some(max_typos))
//!     })
//!     .collect::<Vec<_>>();
//! let mut matcher = Matcher::from_patterns(&patterns, &Config::default());
//! let matches = matcher.match_list(&haystacks);
//! ```
//!
//! # Example: using explicit `Pattern`s
//!
//! If query syntax is not a good fit, build patterns directly and pass them to
//! `Matcher::from_patterns` or `Matcher::new` (if you only have one pattern).
//!
//! ```rust
//! use neo_frizbee::{Config, Matcher, Matching, Pattern, PatternConfig};
//!
//! let patterns = [
//!     Pattern::new("foo", PatternConfig::default()),
//!     Pattern::new("bar", PatternConfig::default().matching(Some(Matching::Prefix))).negated(true),
//! ];
//! let haystacks = ["foo", "barfoo", "foobar"];
//!
//! let mut matcher = Matcher::from_patterns(&patterns, &Config::default());
//! let matches = matcher.match_list(&haystacks);
//! ```
//!
//! # Example: using `FuzzyMatchExt`
//!
//! The iterator API is convenient when chaining with other iterator adapters,
//! but it is slower than matching a full list with `Matcher::match_list`.
//!
//! ```rust
//! use neo_frizbee::{iter::FuzzyMatchExt, Config, radix_sort_matches};
//!
//! let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
//! let mut matches: Vec<_> = haystacks
//!     .iter()
//!     .fuzzy_match("fBr", &Config::default())
//!     .collect();
//! radix_sort_matches(&mut matches);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(all(test, not(feature = "std")))]
compile_error!("frizbee's tests require the `std` feature");

use alloc::{vec, vec::Vec};
use core::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod r#const;
#[cfg(target_arch = "x86_64")]
mod cpuid;
pub mod k_merge;
mod literal;
mod matcher;
mod pattern;
mod prefilter;
mod smith_waterman;
mod sort;

use r#const::*;

pub use r#const::SIMD_CHUNK_BYTES;
pub use matcher::Matcher;
pub use pattern::{Pattern, PatternConfig};
pub use sort::radix_sort_matches;

/// Iterator extension for fuzzy matching
///
/// ```
/// use neo_frizbee::{Config, iter::FuzzyMatchExt};
///
/// let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
/// let matches: Vec<_> = haystacks
///     .iter()
///     .fuzzy_match("fBr", &Config::default())
///     .collect();
/// ```
pub mod iter {
    pub use crate::matcher::{FuzzyMatch, FuzzyMatchExt, FuzzyMatchIndices};
}

/// Matches items in parallel on multiple real threads, resolving each item's
/// haystack bytes through the `resolve` callback, returning a list of
/// [`Match`] values ordered by the configured [`SortStrategy`]. Shorthand for
/// [`Matcher::match_list_parallel_resolved`] when re-using the [`Matcher`]
/// isn't necessary.
///
/// For each item, `resolve` is called with a stack buffer. It should fill the
/// buffer with pointers to [`SIMD_CHUNK_BYTES`]-wide chunks of the haystack
/// (e.g. into an arena) and return `Some((chunk_count, byte_len))`, or `None`
/// to skip the item (e.g. deleted files). This avoids materializing contiguous
/// strings for items whose bytes live in non-contiguous storage.
///
/// `N` is the chunk pointer capacity and must cover the longest haystack:
/// `max_haystack_bytes.div_ceil(SIMD_CHUNK_BYTES)`.
///
/// # Pointer contract
/// See [`Matcher::match_list_resolved_into`].
#[cfg(all(feature = "std", not(target_family = "wasm")))]
pub fn match_list_parallel_resolved<S, T, F, const N: usize>(
    needle: S,
    items: &[T],
    resolve: &F,
    config: &Config,
    threads: usize,
) -> Vec<Match>
where
    S: AsRef<str>,
    T: Sync,
    F: Fn(&T, &mut [*const u8; N]) -> Option<(usize, u16)> + Sync,
{
    Matcher::new(needle.as_ref(), config).match_list_parallel_resolved(items, resolve, threads)
}

/// Result of a fuzzy match, containing the score and index in the haystack
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Match {
    pub score: u16,
    /// Index of the match in the original list of haystacks
    pub index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    pub exact: bool,
    /// Column position (0-based haystack byte offset) where the best alignment
    /// ends. Only populated when the `match_end_col` feature is enabled.
    #[cfg(feature = "match_end_col")]
    pub end_col: u16,
}

impl Match {
    pub fn from_index(index: usize) -> Self {
        Self {
            score: 0,
            index: index as u32,
            exact: false,
            #[cfg(feature = "match_end_col")]
            end_col: 0,
        }
    }
}

impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Match) -> Option<Ordering> {
        Some(core::cmp::Ord::cmp(self, other))
    }
}
impl Ord for Match {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.score as u64)
            .cmp(&(other.score as u64))
            .reverse()
            .then_with(|| self.index.cmp(&other.index))
    }
}
impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}
impl Eq for Match {}

/// Like [`Match`] but includes the indices of the chars in the haystack that
/// matched the needle in reverse order
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MatchIndices {
    pub score: u16,
    /// Index of the match in the original list of haystacks
    pub index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    pub exact: bool,
    /// Indices of the chars in the haystack that matched the needle in reverse
    /// order
    pub indices: Vec<u32>,
}

impl MatchIndices {
    pub fn from_index(index: usize) -> Self {
        Self {
            score: 0,
            index: u32::try_from(index)
                .expect("too many items in haystack, will overflow the u32 index"),
            exact: false,
            indices: vec![],
        }
    }
}

impl PartialOrd for MatchIndices {
    fn partial_cmp(&self, other: &MatchIndices) -> Option<Ordering> {
        Some(core::cmp::Ord::cmp(self, other))
    }
}
impl Ord for MatchIndices {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.score as u64)
            .cmp(&(other.score as u64))
            .reverse()
            .then_with(|| self.index.cmp(&other.index))
    }
}
impl PartialEq for MatchIndices {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}
impl Eq for MatchIndices {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Config {
    /// The maximum number of characters missing from the needle, before an item
    /// in the haystack is filtered out
    pub max_typos: Option<u16>,
    /// Controls how case sensitivity/insensitivity is handled while matching
    #[cfg_attr(feature = "serde", serde(default))]
    pub casing: CaseMatching,
    /// Controls how unicode is handled while matching
    #[cfg_attr(feature = "serde", serde(default))]
    pub unicode: UnicodeMatching,
    /// Selects the matching algorithm: fuzzy (Smith-Waterman) or one of the
    /// literal modes (exact, prefix, suffix, substring). Literal modes
    /// require the needle to appear as a contiguous run of characters and
    /// do not support typos (`max_typos` is ignored).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matching: Matching,
    /// Controls how results are ordered
    #[cfg_attr(feature = "serde", serde(default))]
    pub sort: SortStrategy,
    /// Controls the scoring used by the smith waterman algorithm. Pay close
    /// attention to the documentation for each property, as small changes
    /// can lead to poor matching.
    pub scoring: Scoring,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_typos: Some(0),
            casing: CaseMatching::Smart,
            unicode: UnicodeMatching::Smart,
            matching: Matching::Fuzzy,
            sort: SortStrategy::ScoreThenIndexAsc,
            scoring: Scoring::default(),
        }
    }
}

impl Config {
    /// Sets the matching mode
    pub fn matching(mut self, matching: Matching) -> Self {
        self.matching = matching;
        self
    }

    /// Sets the maximum number of typos allowed
    pub fn max_typos(mut self, max_typos: Option<u16>) -> Self {
        self.max_typos = max_typos;
        self
    }

    /// Sets the casing mode
    pub fn casing(mut self, casing: CaseMatching) -> Self {
        self.casing = casing;
        self
    }

    /// Sets the unicode mode
    pub fn unicode(mut self, unicode: UnicodeMatching) -> Self {
        self.unicode = unicode;
        self
    }

    /// Sets how results are ordered
    pub fn sort(mut self, sort: SortStrategy) -> Self {
        self.sort = sort;
        self
    }

    /// Sets the scoring
    pub fn scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = scoring;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SortStrategy {
    /// Sort by descending score, then ascending haystack index
    #[default]
    ScoreThenIndexAsc,
    /// Sort by descending score, then descending haystack index
    ScoreThenIndexDesc,
    /// Sort by ascending haystack index, preserving input order
    IndexAsc,
    /// Sort by descending haystack index, reversing input order
    IndexDesc,
}

impl SortStrategy {
    pub fn reverse(self) -> Self {
        match self {
            SortStrategy::ScoreThenIndexAsc => SortStrategy::ScoreThenIndexDesc,
            SortStrategy::IndexAsc => SortStrategy::IndexDesc,
            SortStrategy::ScoreThenIndexDesc => SortStrategy::ScoreThenIndexAsc,
            SortStrategy::IndexDesc => SortStrategy::IndexAsc,
        }
    }

    /// Whether the sort strategy matches index (asc) (normal order) or index
    /// (desc) (reverse order).
    ///
    /// When this is `true`, the sort strategy may still sort by score (desc)
    /// first, see [`SortStrategy::is_by_score`].
    pub fn is_reversed(self) -> bool {
        matches!(
            self,
            SortStrategy::IndexDesc | SortStrategy::ScoreThenIndexDesc
        )
    }

    /// Whether the sort strategy matches by score (desc) first
    pub fn is_by_score(self) -> bool {
        matches!(
            self,
            SortStrategy::ScoreThenIndexAsc | SortStrategy::ScoreThenIndexDesc
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CaseMatching {
    /// Ignore case while matching
    Ignore,
    /// Ignore case unless the needle contains uppercase
    #[default]
    Smart,
    /// Require matching bytes to have the same case
    Respect,
}

impl CaseMatching {
    #[inline(always)]
    pub(crate) fn respects_case_for(self, needle: &str) -> bool {
        match self {
            CaseMatching::Ignore => false,
            CaseMatching::Smart => needle.chars().any(char::is_uppercase),
            CaseMatching::Respect => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UnicodeMatching {
    /// Always match against bytes directly
    Ignore,
    /// Ignore unicode unless the needle contains a multi-byte unicode char
    #[default]
    Smart,
    /// Always use expensive unicode Smith Waterman for correctness across
    /// multi-byte unicode chars in the haystack
    Always,
}

impl UnicodeMatching {
    #[inline(always)]
    pub(crate) fn respects_unicode_for(self, needle: &str) -> bool {
        match self {
            UnicodeMatching::Ignore => false,
            UnicodeMatching::Smart => !needle.is_ascii(),
            UnicodeMatching::Always => true,
        }
    }
}

/// Selects the matching algorithm
///
/// [`Matching::Fuzzy`] uses the Smith-Waterman algorithm with typos, gaps and
/// substitutions (default) [`Matching::Exact`] matches the haystack exactly
/// [`Matching::Prefix`] matches the haystack if it starts with the needle
/// [`Matching::Suffix`] matches the haystack if it ends with the needle
/// [`Matching::Substring`] matches the haystack if it contains the needle
///
/// Only the [`Matching::Fuzzy`] mode supports typos
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Matching {
    /// Smith-Waterman fuzzy matching with typos, gaps and substitutions
    /// (default)
    #[default]
    Fuzzy,
    /// The haystack must equal the needle
    Exact,
    /// The haystack must start with the needle
    Prefix,
    /// The haystack must end with the needle
    Suffix,
    /// The needle must appear somewhere in the haystack. When it appears more
    /// than once, the highest-scoring occurrence is used, preferring
    /// earlier matches on tie
    Substring,
}

impl Matching {
    #[inline(always)]
    pub(crate) fn is_fuzzy(self) -> bool {
        matches!(self, Matching::Fuzzy)
    }
}

/// Controls the scoring used by the smith waterman algorithm. Pay close
/// attention to the documentation for each property, as small changes can lead
/// to poor matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Scoring {
    /// Score for a matching character between needle and haystack
    pub match_score: u16,
    /// Penalty for a mismatch (substitution)
    pub mismatch_penalty: u16,
    /// Penalty for opening a gap (deletion/insertion)
    pub gap_open_penalty: u16,
    /// Penalty for extending a gap (deletion/insertion)
    pub gap_extend_penalty: u16,

    /// Bonus for matching the first character of the haystack (e.g. "h" on
    /// "hello_world")
    pub prefix_bonus: u16,
    /// Bonus for matching a capital letter after a lowercase letter
    /// (e.g. "b" on "fooBar" will receive a bonus on "B")
    pub capitalization_bonus: u16,
    /// Bonus for matching the case of the needle (e.g. "WorLd" on "WoRld" will
    /// receive a bonus on "W", "o", "d")
    pub matching_case_bonus: u16,
    /// Bonus for matching the exact needle (e.g. "foo" on "foo" will receive
    /// the bonus)
    pub exact_match_bonus: u16,
    /// Bonus for matching _after_ a delimiter character (e.g. "hw" on
    /// "hello_world", will give a bonus on "w") if "_" is included in the
    /// delimiters string
    pub delimiter_bonus: u16,
}

impl Default for Scoring {
    fn default() -> Self {
        Scoring {
            match_score: MATCH_SCORE,
            mismatch_penalty: MISMATCH_PENALTY,
            gap_open_penalty: GAP_OPEN_PENALTY,
            gap_extend_penalty: GAP_EXTEND_PENALTY,

            prefix_bonus: PREFIX_BONUS,
            capitalization_bonus: CAPITALIZATION_BONUS,
            matching_case_bonus: MATCHING_CASE_BONUS,
            exact_match_bonus: EXACT_MATCH_BONUS,
            delimiter_bonus: DELIMITER_BONUS,
        }
    }
}

impl Scoring {
    /// Needle length up to which scores are guaranteed to fit within the `u16`
    /// score. Longer needles still match, but their scores may saturate at
    /// `u16::MAX`
    pub fn max_needle_len(&self) -> usize {
        let max_per_char = self.match_score.saturating_add(self.max_per_char_bonus());
        // A zero per-char score can never overflow regardless of needle length
        if max_per_char == 0 {
            return usize::MAX;
        }

        // The diagonal transiently holds the score plus the mismatch penalty before
        // subtracting it
        let headroom = u16::MAX
            .saturating_sub(self.max_one_time_bonus())
            .saturating_sub(self.prefix_bonus)
            .saturating_sub(self.exact_match_bonus)
            .saturating_sub(self.mismatch_penalty);
        let max_needle_len = headroom / max_per_char;
        max_needle_len as usize
    }

    /// Max additional score that a needle character can receive, aside from the
    /// match score
    pub(crate) fn max_per_char_bonus(&self) -> u16 {
        let bonus = self.delimiter_bonus.max(self.capitalization_bonus);
        let amortized = bonus
            .div_ceil(2)
            .max(bonus.saturating_sub(self.gap_open_penalty));
        amortized.saturating_add(self.matching_case_bonus)
    }

    /// Max bonus given to a score one time, aside from the prefix or exact
    /// bonuses
    pub(crate) fn max_one_time_bonus(&self) -> u16 {
        let bonus = self.delimiter_bonus.max(self.capitalization_bonus);
        let amortized = bonus
            .div_ceil(2)
            .max(bonus.saturating_sub(self.gap_open_penalty));
        bonus - amortized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_needle_len_default() {
        assert_eq!(Scoring::default().max_needle_len(), 3639);
    }
}
