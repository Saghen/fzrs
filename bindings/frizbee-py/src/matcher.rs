use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyTuple, PyType};

use crate::config::{
    PyScoring, build_config, casing_to_str, matching_to_str, parse_casing, parse_matching,
    parse_sort, parse_unicode, sort_to_str, unicode_to_str,
};
use crate::haystacks::{PyHaystacks, with_haystacks};
use crate::pattern::{PatternArg, PyPattern};

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// Result of a fuzzy match, containing the score and index in the haystack
#[pyclass(
    name = "Match",
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "frizbee"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyMatch {
    /// Score of the match, higher is better
    score: u16,
    /// Index of the match in the original list of haystacks
    index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    exact: bool,
}

#[pymethods]
impl PyMatch {
    fn __repr__(&self) -> String {
        format!(
            "Match(score={}, index={}, exact={})",
            self.score,
            self.index,
            py_bool(self.exact)
        )
    }
}

impl From<frizbee::Match> for PyMatch {
    fn from(matched: frizbee::Match) -> Self {
        Self {
            score: matched.score,
            index: matched.index,
            exact: matched.exact,
        }
    }
}

/// Like [`PyMatch`] but includes the indices of the characters in the haystack
/// that matched the needle
#[pyclass(
    name = "MatchIndices",
    frozen,
    eq,
    hash,
    skip_from_py_object,
    module = "frizbee"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyMatchIndices {
    /// Score of the match, higher is better
    #[pyo3(get)]
    score: u16,
    /// Index of the match in the original list of haystacks
    #[pyo3(get)]
    index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    #[pyo3(get)]
    exact: bool,
    /// Ascending Python character indices in the haystack that matched the
    /// needle
    indices: Vec<u32>,
}

#[pymethods]
impl PyMatchIndices {
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.indices)
    }

    fn __repr__(&self) -> String {
        let indices = match self.indices.as_slice() {
            [] => "()".to_owned(),
            [index] => format!("({index},)"),
            indices => format!(
                "({})",
                indices
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        format!(
            "MatchIndices(score={}, index={}, exact={}, indices={})",
            self.score,
            self.index,
            py_bool(self.exact),
            indices
        )
    }
}

impl PyMatchIndices {
    /// Converts the core's reverse UTF-8 byte offsets into ascending Python
    /// character indices
    fn from_core(matched: frizbee::MatchIndices, haystack: &str) -> Self {
        let mut byte_indices = matched.indices;
        byte_indices.sort_unstable();

        let mut indices = Vec::with_capacity(byte_indices.len());
        let mut byte_indices = byte_indices.into_iter().peekable();
        for (char_index, (start, character)) in haystack.char_indices().enumerate() {
            let end = start + character.len_utf8();
            let char_index = char_index as u32;

            while let Some(&byte_index) = byte_indices.peek() {
                let byte_index = byte_index as usize;
                if byte_index >= end {
                    break;
                }
                byte_indices.next();
                if byte_index >= start && indices.last() != Some(&char_index) {
                    indices.push(char_index);
                }
            }

            if byte_indices.peek().is_none() {
                break;
            }
        }

        Self {
            score: matched.score,
            index: matched.index,
            exact: matched.exact,
            indices,
        }
    }
}

#[derive(FromPyObject)]
enum PatternsArg {
    One(PatternArg),
    Many(Vec<PatternArg>),
}

impl PatternsArg {
    fn into_patterns(self) -> Vec<frizbee::Pattern> {
        match self {
            Self::One(pattern) => vec![pattern.into()],
            Self::Many(patterns) => patterns.into_iter().map(Into::into).collect(),
        }
    }
}

/// Primary entrypoint for fuzzy matching
///
/// `Matcher` compiles the patterns once, allocates memory for the Smith
/// Waterman matrix, and reuses the selected SIMD backend
///
/// Ideally, only construct these at most once per list. They're cheap to
/// construct, but end up being expensive if you construct them for each item in
/// your list.
#[pyclass(name = "Matcher", module = "frizbee")]
pub(crate) struct PyMatcher {
    // Compiled SIMD backends require 64-byte alignment (AVX-512 vectors), which CPython does not
    // guarantee for the pyclass payload itself, so box
    inner: Box<frizbee::Matcher>,
    /// Truncates results after sorting; `None` means unlimited
    limit: Option<u32>,
}

impl PyMatcher {
    fn truncate<T>(&self, matches: &mut Vec<T>) {
        if let Some(limit) = self.limit {
            matches.truncate(limit as usize);
        }
    }

    fn match_indices_impl<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<PyMatchIndices> {
        let mut matches = self.inner.match_list_indices(haystacks);
        self.truncate(&mut matches);
        matches
            .into_iter()
            .map(|matched| {
                let haystack = haystacks[matched.index as usize].as_ref();
                PyMatchIndices::from_core(matched, haystack)
            })
            .collect()
    }
}

#[pymethods]
impl PyMatcher {
    /// Creates a matcher from one or more patterns (string or `Pattern`),
    /// matched independently. Strings convert into patterns that match
    /// literally. Use `from_query` for query syntax. A haystack matches when
    /// all of the patterns match, where the score is the sum of each pattern's
    /// score
    #[new]
    #[pyo3(signature = (patterns, *, max_typos = frizbee::Config::default().max_typos,
        limit = None, casing = None, unicode = None, matching = None, sort = None,
        scoring = None), text_signature = "(patterns, *, max_typos=0, limit=None, casing=None, unicode=None, matching=None, sort=None, scoring=None)")]
    #[allow(clippy::too_many_arguments)]
    fn new(
        patterns: PatternsArg,
        max_typos: Option<u16>,
        limit: Option<u32>,
        casing: Option<&str>,
        unicode: Option<&str>,
        matching: Option<&str>,
        sort: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<Self> {
        let patterns = patterns.into_patterns();
        let config = build_config(max_typos, casing, unicode, matching, sort, scoring)?;
        Ok(Self {
            inner: Box::new(frizbee::Matcher::from_patterns(&patterns, &config)),
            limit,
        })
    }

    /// Parses whitespace-separated query atoms, where special syntax changes
    /// each atom's matching mode:
    ///
    /// - `foo` - defers to matcher config, which defaults to fuzzy
    /// - `^foo` - prefix
    /// - `foo$` - suffix
    /// - `'foo` - substring
    /// - `^foo$` - exact
    /// - `!foo` - negated, substring unless combined with the syntax above
    ///
    /// Any special character can be escaped with a backslash, e.g. `\!foo`,
    /// `\^foo`, `foo\$` or `\'foo` match the literal leading/trailing
    /// character, and `foo\ bar` matches the literal space.
    #[classmethod]
    #[pyo3(signature = (query, *, max_typos = frizbee::Config::default().max_typos,
        limit = None, casing = None, unicode = None, matching = None, sort = None,
        scoring = None), text_signature = "($cls, query, *, max_typos=0, limit=None, casing=None, unicode=None, matching=None, sort=None, scoring=None)")]
    #[allow(clippy::too_many_arguments)]
    fn from_query(
        _cls: &Bound<'_, PyType>,
        query: &str,
        max_typos: Option<u16>,
        limit: Option<u32>,
        casing: Option<&str>,
        unicode: Option<&str>,
        matching: Option<&str>,
        sort: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<Self> {
        let config = build_config(max_typos, casing, unicode, matching, sort, scoring)?;
        Ok(Self {
            inner: Box::new(frizbee::Matcher::from_query(query, &config)),
            limit,
        })
    }

    /// Matches a list of haystacks
    ///
    /// When `threads` is set, matches in parallel on multiple real threads. If
    /// `threads == 0`, the matcher will default to available CPU cores - 2.
    ///
    /// This API provides the most performant path when matching on lists.
    /// Ordinary iterables are copied into Rust memory before parallel matching;
    /// matching then runs with the GIL released for every input type.
    /// Sequential matching releases the GIL when passed a `Haystacks`.
    #[pyo3(signature = (haystacks, *, threads = None))]
    fn r#match(
        &mut self,
        py: Python<'_>,
        haystacks: &Bound<'_, PyAny>,
        threads: Option<usize>,
    ) -> PyResult<Vec<PyMatch>> {
        let mut matches = match threads {
            Some(threads) => {
                let owned;
                let temporary;
                let strs = if let Ok(haystacks) = haystacks.cast::<PyHaystacks>() {
                    owned = haystacks.borrow();
                    owned.as_strs()
                } else {
                    temporary = PyHaystacks::from_iterable(haystacks)?;
                    temporary.as_strs()
                };
                let inner = &mut self.inner;
                py.detach(move || inner.match_list_parallel(&strs, threads))
            }
            None => {
                if let Ok(haystacks) = haystacks.cast::<PyHaystacks>() {
                    let haystacks = haystacks.borrow();
                    let strs = haystacks.as_strs();
                    let inner = &mut self.inner;
                    py.detach(move || inner.match_list(&strs))
                } else {
                    with_haystacks(haystacks, |strs| Ok(self.inner.match_list(strs)))?
                }
            }
        };
        self.truncate(&mut matches);
        Ok(matches.into_iter().map(Into::into).collect())
    }

    /// Matches a list of haystacks, returning a list of `MatchIndices` which
    /// are equivalent to `Match` except they include the indices of the matched
    /// characters in the haystack.
    ///
    /// This API has not been optimized for performance, and should only be used
    /// on small lists or after matching a list of haystacks with
    /// `Matcher::match`. Useful for displaying matched indices in the UI.
    ///
    /// The GIL is released while matching if passed a `Haystacks`.
    fn match_indices(
        &mut self,
        py: Python<'_>,
        haystacks: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyMatchIndices>> {
        if let Ok(haystacks) = haystacks.cast::<PyHaystacks>() {
            let haystacks = haystacks.borrow();
            let strs = haystacks.as_strs();
            Ok(py.detach(move || self.match_indices_impl(&strs)))
        } else {
            with_haystacks(haystacks, |strs| Ok(self.match_indices_impl(strs)))
        }
    }

    /// Matches a single haystack, returning its `Match` if it passes. This API
    /// performs much slower than the `match` API with `Haystacks`
    #[pyo3(signature = (haystack, index = 0))]
    fn match_one(&mut self, haystack: &str, index: u32) -> Option<PyMatch> {
        self.inner.match_one(haystack, index).map(Into::into)
    }

    /// Like `match_one` but includes the indices of the chars in the haystack
    /// that matched the needle. Useful for displaying matched indices in the UI
    #[pyo3(signature = (haystack, index = 0))]
    fn match_one_indices(&mut self, haystack: &str, index: u32) -> Option<PyMatchIndices> {
        self.inner
            .match_one_indices(haystack, index)
            .map(|matched| PyMatchIndices::from_core(matched, haystack))
    }

    /// Updates any combination of patterns, matcher options, and result limit.
    /// Omitted values remain unchanged. `None` means unlimited for `max_typos`
    /// and `limit`
    #[pyo3(signature = (*, patterns = None,
        max_typos = Python::attach(|py| py.Ellipsis()),
        limit = Python::attach(|py| py.Ellipsis()),
        casing = None, unicode = None, matching = None, sort = None, scoring = None),
        text_signature = "($self, /, *, patterns=None, max_typos=..., limit=..., casing=None, unicode=None, matching=None, sort=None, scoring=None)")]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        py: Python<'_>,
        patterns: Option<PatternsArg>,
        max_typos: Py<PyAny>,
        limit: Py<PyAny>,
        casing: Option<&str>,
        unicode: Option<&str>,
        matching: Option<&str>,
        sort: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<()> {
        let changed = |value: &Py<PyAny>| !value.bind(py).is_instance_of::<PyEllipsis>();
        let limit_changed = changed(&limit);
        let next_limit = if limit_changed {
            limit.extract::<Option<u32>>(py)?
        } else {
            self.limit
        };

        let config_changed = changed(&max_typos)
            || casing.is_some()
            || unicode.is_some()
            || matching.is_some()
            || sort.is_some()
            || scoring.is_some();
        let mut next_config = *self.inner.config();
        if changed(&max_typos) {
            next_config.max_typos = max_typos.extract::<Option<u16>>(py)?;
        }
        if let Some(casing) = casing {
            next_config.casing = parse_casing(casing)?;
        }
        if let Some(unicode) = unicode {
            next_config.unicode = parse_unicode(unicode)?;
        }
        if let Some(matching) = matching {
            next_config.matching = parse_matching(matching)?;
        }
        if let Some(sort) = sort {
            next_config.sort = parse_sort(sort)?;
        }
        if let Some(scoring) = scoring {
            next_config.scoring = scoring.into();
        }

        if patterns.is_some() || config_changed {
            let patterns = patterns.map_or_else(
                || self.inner.patterns().to_vec(),
                PatternsArg::into_patterns,
            );
            *self.inner = frizbee::Matcher::from_patterns(&patterns, &next_config);
        }
        self.limit = next_limit;
        Ok(())
    }

    /// The current patterns returned as copies (mutate them freely)
    #[getter]
    fn patterns(&self) -> Vec<PyPattern> {
        self.inner
            .patterns()
            .iter()
            .map(|pattern| PyPattern {
                inner: pattern.clone(),
            })
            .collect()
    }

    #[getter]
    fn max_typos(&self) -> Option<u16> {
        self.inner.config().max_typos
    }

    /// Max matches returned from the `match` APIs. `None` means unlimited
    #[getter]
    fn limit(&self) -> Option<u32> {
        self.limit
    }

    #[getter]
    fn casing(&self) -> &'static str {
        casing_to_str(self.inner.config().casing)
    }

    #[getter]
    fn unicode(&self) -> &'static str {
        unicode_to_str(self.inner.config().unicode)
    }

    #[getter]
    fn matching(&self) -> &'static str {
        matching_to_str(self.inner.config().matching)
    }

    #[getter]
    fn sort(&self) -> &'static str {
        sort_to_str(self.inner.config().sort)
    }

    #[getter]
    fn scoring(&self) -> PyScoring {
        self.inner.config().scoring.into()
    }
}
