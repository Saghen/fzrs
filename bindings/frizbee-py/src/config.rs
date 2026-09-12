use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use frizbee::{CaseMatching, Config, Matching, SortStrategy, UnicodeMatching};

/// Generates the `parse_*`/`*_to_str` pair mapping a core enum to its Python
/// string literals, with a `ValueError` listing the valid values on a miss
macro_rules! str_enum {
    ($parse:ident, $to_str:ident, $ty:ty, $name:literal, { $($str:literal => $variant:path),+ $(,)? }) => {
        pub(crate) fn $parse(value: &str) -> PyResult<$ty> {
            match value {
                $($str => Ok($variant),)+
                other => Err(PyValueError::new_err(format!(
                    concat!("invalid ", $name, " {:?}; expected one of ", str_enum!(@list $($str),+)),
                    other
                ))),
            }
        }

        pub(crate) fn $to_str(value: $ty) -> &'static str {
            match value {
                $($variant => $str,)+
            }
        }
    };
    (@list $first:literal $(, $rest:literal)*) => {
        concat!("'", $first, "'" $(, ", '", $rest, "'")*)
    };
}

str_enum!(parse_casing, casing_to_str, CaseMatching, "casing", {
    "ignore" => CaseMatching::Ignore,
    "smart" => CaseMatching::Smart,
    "respect" => CaseMatching::Respect,
});

str_enum!(parse_unicode, unicode_to_str, UnicodeMatching, "unicode", {
    "ignore" => UnicodeMatching::Ignore,
    "smart" => UnicodeMatching::Smart,
    "always" => UnicodeMatching::Always,
});

str_enum!(parse_matching, matching_to_str, Matching, "matching", {
    "fuzzy" => Matching::Fuzzy,
    "exact" => Matching::Exact,
    "prefix" => Matching::Prefix,
    "suffix" => Matching::Suffix,
    "substring" => Matching::Substring,
});

str_enum!(parse_sort, sort_to_str, SortStrategy, "sort", {
    "score_then_index_asc" => SortStrategy::ScoreThenIndexAsc,
    "score_then_index_desc" => SortStrategy::ScoreThenIndexDesc,
    "index_asc" => SortStrategy::IndexAsc,
    "index_desc" => SortStrategy::IndexDesc,
    "unsorted" => SortStrategy::Unsorted,
});

/// Controls the scoring used by the smith waterman algorithm.
///
/// Mirrors the core defaults when a field is omitted. Pay close attention to
/// the documentation for each property, as small changes can lead to poor
/// matching.
#[pyclass(
    name = "Scoring",
    frozen,
    get_all,
    eq,
    hash,
    from_py_object,
    module = "frizbee"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyScoring {
    /// Score for a matching character between needle and haystack
    match_score: u16,
    /// Penalty for a mismatch (substitution)
    mismatch_penalty: u16,
    /// Penalty for opening a gap (deletion/insertion)
    gap_open_penalty: u16,
    /// Penalty for extending a gap (deletion/insertion)
    gap_extend_penalty: u16,
    /// Bonus for matching the first character of the haystack (e.g. "h" on
    /// "hello_world")
    prefix_bonus: u16,
    /// Bonus for matching a capital letter after a lowercase letter
    /// (e.g. "b" on "fooBar" will receive a bonus on "B")
    capitalization_bonus: u16,
    /// Bonus for matching the case of the needle (e.g. "WorLd" on "WoRld" will
    /// receive a bonus on "W", "o", "d")
    matching_case_bonus: u16,
    /// Bonus for matching the exact needle (e.g. "foo" on "foo" will receive
    /// the bonus)
    exact_match_bonus: u16,
    /// Bonus for matching _after_ a delimiter character (e.g. "hw" on
    /// "hello_world" will give a bonus on "w")
    delimiter_bonus: u16,
}

impl From<PyScoring> for frizbee::Scoring {
    fn from(s: PyScoring) -> Self {
        frizbee::Scoring {
            match_score: s.match_score,
            mismatch_penalty: s.mismatch_penalty,
            gap_open_penalty: s.gap_open_penalty,
            gap_extend_penalty: s.gap_extend_penalty,
            prefix_bonus: s.prefix_bonus,
            capitalization_bonus: s.capitalization_bonus,
            matching_case_bonus: s.matching_case_bonus,
            exact_match_bonus: s.exact_match_bonus,
            delimiter_bonus: s.delimiter_bonus,
        }
    }
}

impl From<frizbee::Scoring> for PyScoring {
    fn from(s: frizbee::Scoring) -> Self {
        Self {
            match_score: s.match_score,
            mismatch_penalty: s.mismatch_penalty,
            gap_open_penalty: s.gap_open_penalty,
            gap_extend_penalty: s.gap_extend_penalty,
            prefix_bonus: s.prefix_bonus,
            capitalization_bonus: s.capitalization_bonus,
            matching_case_bonus: s.matching_case_bonus,
            exact_match_bonus: s.exact_match_bonus,
            delimiter_bonus: s.delimiter_bonus,
        }
    }
}

#[pymethods]
impl PyScoring {
    #[new]
    #[pyo3(signature = (*, match_score = None, mismatch_penalty = None, gap_open_penalty = None,
        gap_extend_penalty = None, prefix_bonus = None, capitalization_bonus = None,
        matching_case_bonus = None, exact_match_bonus = None, delimiter_bonus = None),
        text_signature = "(*, match_score=12, mismatch_penalty=6, gap_open_penalty=5, gap_extend_penalty=1, prefix_bonus=12, capitalization_bonus=4, matching_case_bonus=4, exact_match_bonus=8, delimiter_bonus=4)")]
    #[allow(clippy::too_many_arguments)]
    fn new(
        match_score: Option<u16>,
        mismatch_penalty: Option<u16>,
        gap_open_penalty: Option<u16>,
        gap_extend_penalty: Option<u16>,
        prefix_bonus: Option<u16>,
        capitalization_bonus: Option<u16>,
        matching_case_bonus: Option<u16>,
        exact_match_bonus: Option<u16>,
        delimiter_bonus: Option<u16>,
    ) -> Self {
        let default = frizbee::Scoring::default();
        Self {
            match_score: match_score.unwrap_or(default.match_score),
            mismatch_penalty: mismatch_penalty.unwrap_or(default.mismatch_penalty),
            gap_open_penalty: gap_open_penalty.unwrap_or(default.gap_open_penalty),
            gap_extend_penalty: gap_extend_penalty.unwrap_or(default.gap_extend_penalty),
            prefix_bonus: prefix_bonus.unwrap_or(default.prefix_bonus),
            capitalization_bonus: capitalization_bonus.unwrap_or(default.capitalization_bonus),
            matching_case_bonus: matching_case_bonus.unwrap_or(default.matching_case_bonus),
            exact_match_bonus: exact_match_bonus.unwrap_or(default.exact_match_bonus),
            delimiter_bonus: delimiter_bonus.unwrap_or(default.delimiter_bonus),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Scoring(match_score={}, mismatch_penalty={}, gap_open_penalty={}, \
             gap_extend_penalty={}, prefix_bonus={}, capitalization_bonus={}, \
             matching_case_bonus={}, exact_match_bonus={}, delimiter_bonus={})",
            self.match_score,
            self.mismatch_penalty,
            self.gap_open_penalty,
            self.gap_extend_penalty,
            self.prefix_bonus,
            self.capitalization_bonus,
            self.matching_case_bonus,
            self.exact_match_bonus,
            self.delimiter_bonus,
        )
    }

    /// Needle length up to which scores are guaranteed to fit within the 16-bit
    /// score.
    ///
    /// Longer needles still match, but their scores may saturate at `0xffff`.
    #[getter]
    fn max_needle_len(&self) -> usize {
        frizbee::Scoring::from(self.clone()).max_needle_len()
    }
}

/// Builds a core config from the kwargs shared by every `Matcher` constructor
/// and applies defaults
pub(crate) fn build_config(
    max_typos: Option<u16>,
    casing: Option<&str>,
    unicode: Option<&str>,
    matching: Option<&str>,
    sort: Option<&str>,
    scoring: Option<PyScoring>,
) -> PyResult<Config> {
    let default = Config::default();
    Ok(Config {
        max_typos,
        casing: casing
            .map(parse_casing)
            .transpose()?
            .unwrap_or(default.casing),
        unicode: unicode
            .map(parse_unicode)
            .transpose()?
            .unwrap_or(default.unicode),
        matching: matching
            .map(parse_matching)
            .transpose()?
            .unwrap_or(default.matching),
        sort: sort.map(parse_sort).transpose()?.unwrap_or(default.sort),
        scoring: scoring.map_or(default.scoring, Into::into),
    })
}
