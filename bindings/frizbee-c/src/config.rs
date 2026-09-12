use frizbee::{CaseMatching, Config, Matching, Scoring, SortStrategy, UnicodeMatching};

use std::ffi::c_char;

/// A borrowed string: `len` bytes at `ptr`, not necessarily null terminated.
/// The bytes MUST be valid UTF-8 as this is not validated.
/// `ptr` may be NULL only when `len` is 0.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct frizbee_str_t {
    pub ptr: *const c_char,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum frizbee_case_matching_t {
    /// Ignore case while matching
    FRIZBEE_CASE_IGNORE = 0,
    /// Ignore case unless the needle contains uppercase (default)
    FRIZBEE_CASE_SMART,
    /// Require matching bytes to have the same case
    FRIZBEE_CASE_RESPECT,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum frizbee_unicode_matching_t {
    /// Always match against bytes directly
    FRIZBEE_UNICODE_IGNORE = 0,
    /// Ignore unicode unless the needle contains a multi-byte unicode char
    /// (default)
    FRIZBEE_UNICODE_SMART,
    /// Always use expensive unicode Smith Waterman for correctness across
    /// multi-byte unicode chars in the haystack
    FRIZBEE_UNICODE_ALWAYS,
}

/// Selects the matching algorithm
///
/// - `FRIZBEE_MATCHING_FUZZY` (default) uses the Smith-Waterman algorithm with
///   typos, gaps and substitutions
/// - `FRIZBEE_MATCHING_EXACT` matches the haystack exactly
/// - `FRIZBEE_MATCHING_PREFIX` matches the haystack if it starts with the
///   needle
/// - `FRIZBEE_MATCHING_SUFFIX` matches the haystack if it ends with the needle
/// - `FRIZBEE_MATCHING_SUBSTRING` matches the haystack if it contains the
///   needle
///
/// Only the `FRIZBEE_MATCHING_FUZZY` mode supports typos
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum frizbee_matching_t {
    /// Smith-Waterman fuzzy matching with typos, gaps and substitutions
    /// (default)
    FRIZBEE_MATCHING_FUZZY = 0,
    /// The haystack must equal the needle
    FRIZBEE_MATCHING_EXACT,
    /// The haystack must start with the needle
    FRIZBEE_MATCHING_PREFIX,
    /// The haystack must end with the needle
    FRIZBEE_MATCHING_SUFFIX,
    /// The needle must appear somewhere in the haystack. When it appears more
    /// than once, the highest-scoring occurrence is used, preferring
    /// earlier matches on tie
    FRIZBEE_MATCHING_SUBSTRING,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum frizbee_sort_strategy_t {
    /// Sort by descending score, then ascending haystack index (default)
    FRIZBEE_SORT_SCORE_THEN_INDEX_ASC = 0,
    /// Sort by descending score, then descending haystack index
    FRIZBEE_SORT_SCORE_THEN_INDEX_DESC,
    /// Sort by ascending haystack index, preserving input order
    FRIZBEE_SORT_INDEX_ASC,
    /// Sort by descending haystack index, reversing input order
    FRIZBEE_SORT_INDEX_DESC,
    /// No ordering guarantee (cheapest); for callers that re-sort themselves
    FRIZBEE_SORT_UNSORTED,
}

/// Controls the scoring used by the smith waterman algorithm. Pay close
/// attention to the documentation for each property, as small changes can lead
/// to poor matching.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct frizbee_scoring_t {
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
    /// "hello_world" will give a bonus on "w")
    pub delimiter_bonus: u16,
}

/// Obtain defaults from `frizbee_config_default`
///
/// Any value outside the named constants falls back to that field's default
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct frizbee_config_t {
    /// The maximum number of characters missing from the needle, before an item
    /// in the haystack is filtered out. Negative means unlimited typos
    pub max_typos: i32,
    /// Controls how case sensitivity/insensitivity is handled while matching.
    /// One of the `frizbee_case_matching_t` values; anything else falls back to
    /// `FRIZBEE_CASE_SMART`
    pub casing: i32,
    /// Controls how unicode is handled while matching
    ///
    /// One of the `frizbee_unicode_matching_t` values. Anything else falls back
    /// to `FRIZBEE_UNICODE_SMART`.
    pub unicode: i32,
    /// Selects the matching algorithm: fuzzy (Smith-Waterman) or one of the
    /// literal modes (exact, prefix, suffix, substring). Literal modes
    /// require the needle to appear as a contiguous run of characters and
    /// do not support typos (`max_typos` is ignored).
    ///
    /// One of the `frizbee_matching_t` values. Anything else falls back to
    /// `FRIZBEE_MATCHING_FUZZY`.
    pub matching: i32,
    /// Controls how results are ordered
    ///
    /// One of the `frizbee_sort_strategy_t` values. Anything else falls back to
    /// `FRIZBEE_SORT_SCORE_THEN_INDEX_ASC`.
    pub sort: i32,
    /// Controls the scoring used by the smith waterman algorithm. Pay close
    /// attention to the documentation for each property, as small changes can
    /// lead to poor matching.
    pub scoring: frizbee_scoring_t,
}

/// Generates the private `*_to_core`/`*_from_core` conversions for enums
///
/// Out-of-range values fallback to the default
macro_rules! convert_enum {
    ($c:ident => $core:ident, $to_core:ident, $from_core:ident,
     { $($c_variant:ident => $core_variant:ident),+ $(,)? }) => {
        fn $to_core(value: i32) -> $core {
            $(if value == $c::$c_variant as i32 {
                return $core::$core_variant;
            })+
            $core::default()
        }

        fn $from_core(value: $core) -> i32 {
            match value {
                $($core::$core_variant => $c::$c_variant as i32,)+
            }
        }
    };
}

convert_enum!(frizbee_case_matching_t => CaseMatching, case_matching_to_core, case_matching_from_core, {
    FRIZBEE_CASE_IGNORE => Ignore,
    FRIZBEE_CASE_SMART => Smart,
    FRIZBEE_CASE_RESPECT => Respect,
});

convert_enum!(frizbee_unicode_matching_t => UnicodeMatching, unicode_matching_to_core, unicode_matching_from_core, {
    FRIZBEE_UNICODE_IGNORE => Ignore,
    FRIZBEE_UNICODE_SMART => Smart,
    FRIZBEE_UNICODE_ALWAYS => Always,
});

convert_enum!(frizbee_matching_t => Matching, matching_to_core, matching_from_core, {
    FRIZBEE_MATCHING_FUZZY => Fuzzy,
    FRIZBEE_MATCHING_EXACT => Exact,
    FRIZBEE_MATCHING_PREFIX => Prefix,
    FRIZBEE_MATCHING_SUFFIX => Suffix,
    FRIZBEE_MATCHING_SUBSTRING => Substring,
});

convert_enum!(frizbee_sort_strategy_t => SortStrategy, sort_strategy_to_core, sort_strategy_from_core, {
    FRIZBEE_SORT_SCORE_THEN_INDEX_ASC => ScoreThenIndexAsc,
    FRIZBEE_SORT_SCORE_THEN_INDEX_DESC => ScoreThenIndexDesc,
    FRIZBEE_SORT_INDEX_ASC => IndexAsc,
    FRIZBEE_SORT_INDEX_DESC => IndexDesc,
    FRIZBEE_SORT_UNSORTED => Unsorted,
});

fn scoring_to_core(scoring: &frizbee_scoring_t) -> Scoring {
    Scoring {
        match_score: scoring.match_score,
        mismatch_penalty: scoring.mismatch_penalty,
        gap_open_penalty: scoring.gap_open_penalty,
        gap_extend_penalty: scoring.gap_extend_penalty,
        prefix_bonus: scoring.prefix_bonus,
        capitalization_bonus: scoring.capitalization_bonus,
        matching_case_bonus: scoring.matching_case_bonus,
        exact_match_bonus: scoring.exact_match_bonus,
        delimiter_bonus: scoring.delimiter_bonus,
    }
}

fn scoring_from_core(scoring: &Scoring) -> frizbee_scoring_t {
    frizbee_scoring_t {
        match_score: scoring.match_score,
        mismatch_penalty: scoring.mismatch_penalty,
        gap_open_penalty: scoring.gap_open_penalty,
        gap_extend_penalty: scoring.gap_extend_penalty,
        prefix_bonus: scoring.prefix_bonus,
        capitalization_bonus: scoring.capitalization_bonus,
        matching_case_bonus: scoring.matching_case_bonus,
        exact_match_bonus: scoring.exact_match_bonus,
        delimiter_bonus: scoring.delimiter_bonus,
    }
}

fn config_to_core(config: &frizbee_config_t) -> Config {
    Config {
        // Negative means unlimited
        max_typos: (config.max_typos >= 0).then(|| config.max_typos.min(u16::MAX.into()) as u16),
        casing: case_matching_to_core(config.casing),
        unicode: unicode_matching_to_core(config.unicode),
        matching: matching_to_core(config.matching),
        sort: sort_strategy_to_core(config.sort),
        scoring: scoring_to_core(&config.scoring),
    }
}

/// SAFETY: when non-NULL, `config` must point to a valid `frizbee_config_t`
pub(crate) unsafe fn config_or_default(config: *const frizbee_config_t) -> Config {
    if config.is_null() {
        Config::default()
    } else {
        config_to_core(unsafe { &*config })
    }
}

fn config_from_core(config: &Config) -> frizbee_config_t {
    frizbee_config_t {
        max_typos: config.max_typos.map(i32::from).unwrap_or(-1),
        casing: case_matching_from_core(config.casing),
        unicode: unicode_matching_from_core(config.unicode),
        matching: matching_from_core(config.matching),
        sort: sort_strategy_from_core(config.sort),
        scoring: scoring_from_core(&config.scoring),
    }
}

/// Returns the default configuration, mirroring `frizbee::Config::default()`
#[unsafe(no_mangle)]
pub extern "C" fn frizbee_config_default() -> frizbee_config_t {
    config_from_core(&Config::default())
}

/// Needle length up to which scores are guaranteed to fit within the `uint16_t`
/// score. Longer needles still match, but their scores may saturate at
/// `UINT16_MAX`. If `scoring` is NULL, returns the limit for default scoring.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_scoring_max_needle_len(
    scoring: *const frizbee_scoring_t,
) -> usize {
    if scoring.is_null() {
        Scoring::default().max_needle_len()
    } else {
        scoring_to_core(unsafe { &*scoring }).max_needle_len()
    }
}
