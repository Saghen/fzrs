use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{CaseMatching, Config, Matching, Scoring, UnicodeMatching};

/// A single pattern to match, parsed from syntax like `!^foo`
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pattern {
    /// Raw atom text, e.g. `!^foo`
    pub pattern: String,
    /// Haystacks matching this atom are excluded
    pub negated: bool,
    /// Text to match with the syntax stripped, e.g. `foo`
    pub needle: String,
    /// Configuration for this pattern
    pub config: PatternConfig,
}

/// Matches the needle literally, without parsing any syntax, using the matching
/// mode from [`crate::Config::matching`]
impl From<&str> for Pattern {
    fn from(needle: &str) -> Self {
        Pattern::new(needle, PatternConfig::default())
    }
}

impl From<String> for Pattern {
    fn from(needle: String) -> Self {
        needle.as_str().into()
    }
}

impl From<&String> for Pattern {
    fn from(needle: &String) -> Self {
        needle.as_str().into()
    }
}

impl Pattern {
    /// Creates a pattern that matches the needle literally, without parsing any
    /// syntax. Use [`Pattern::negated`] to exclude matching haystacks.
    pub fn new(needle: &str, config: PatternConfig) -> Self {
        Self {
            pattern: needle.to_string(),
            negated: false,
            needle: needle.to_string(),
            config,
        }
    }

    /// Sets whether haystacks matching this pattern are excluded
    pub fn negated(mut self, negated: bool) -> Self {
        self.negated = negated;
        self
    }

    /// Overrides [`crate::Config::matching`] for this pattern (see
    /// [`PatternConfig::matching`])
    pub fn matching(mut self, matching: Option<Matching>) -> Self {
        self.config = self.config.matching(matching);
        self
    }

    /// Overrides [`crate::Config::max_typos`] for this pattern (see
    /// [`PatternConfig::max_typos`])
    pub fn max_typos(mut self, max_typos: Option<u16>) -> Self {
        self.config = self.config.max_typos(max_typos);
        self
    }

    /// Overrides [`crate::Config::casing`] for this pattern (see
    /// [`PatternConfig::casing`])
    pub fn casing(mut self, casing: Option<CaseMatching>) -> Self {
        self.config = self.config.casing(casing);
        self
    }

    /// Overrides [`crate::Config::unicode`] for this pattern (see
    /// [`PatternConfig::unicode`])
    pub fn unicode(mut self, unicode: Option<UnicodeMatching>) -> Self {
        self.config = self.config.unicode(unicode);
        self
    }

    /// Overrides [`crate::Config::scoring`] for this pattern (see
    /// [`PatternConfig::scoring`])
    pub fn scoring(mut self, scoring: Option<Scoring>) -> Self {
        self.config = self.config.scoring(scoring);
        self
    }

    /// Parses a single query atom, where special syntax changes the matching
    /// mode:
    ///
    /// - `foo` - `None` (defers to [`crate::Config::matching`])
    /// - `^foo` - [`Matching::Prefix`]
    /// - `foo$` - [`Matching::Suffix`]
    /// - `'foo` - [`Matching::Substring`]
    /// - `^foo$` - [`Matching::Exact`]
    /// - `!foo` - negated, [`Matching::Substring`] unless combined with the
    ///   syntax above
    ///
    /// Any special character can be escaped with a backslash, e.g. `\!foo`,
    /// `\^foo`, `foo\$` or `\'foo` match the literal leading/trailing
    /// character, and `foo\ bar` matches the literal space.
    pub fn parse(atom: &str) -> Self {
        // Tokenize chars, marking whether they're esacped or not
        let mut tokens: Vec<(char, bool)> = Vec::with_capacity(atom.len());
        let mut chars = atom.chars();
        while let Some(c) = chars.next() {
            match if c == '\\' { chars.next() } else { None } {
                Some(escaped) => tokens.push((escaped, true)),
                None => tokens.push((c, false)),
            }
        }

        // Pops the first/last token if it's the unescaped operator `op`
        fn strip_first(tokens: &mut &[(char, bool)], op: char) -> bool {
            if let Some((&(c, false), rest)) = tokens.split_first()
                && c == op
            {
                *tokens = rest;
                return true;
            }
            false
        }
        fn strip_last(tokens: &mut &[(char, bool)], op: char) -> bool {
            if let Some((&(c, false), rest)) = tokens.split_last()
                && c == op
            {
                *tokens = rest;
                return true;
            }
            false
        }

        let mut rest = tokens.as_slice();
        let negated = strip_first(&mut rest, '!');
        let prefix = strip_first(&mut rest, '^');
        let substring = !prefix && strip_first(&mut rest, '\'');
        let suffix = strip_last(&mut rest, '$');

        // Escaped special characters collapse to the literal character; the backslash
        // is kept before anything else (including another backslash)
        let is_special = |c: char| matches!(c, '!' | '^' | '\'' | '$') || c.is_whitespace();
        let mut needle = String::with_capacity(atom.len());
        for &(c, escaped) in rest {
            if escaped && !is_special(c) {
                needle.push('\\');
            }
            needle.push(c);
        }

        let matching = match (prefix, suffix, substring) {
            (true, true, _) => Some(Matching::Exact),
            (true, false, _) => Some(Matching::Prefix),
            (false, true, _) => Some(Matching::Suffix),
            (false, false, true) => Some(Matching::Substring),
            // Negating a fuzzy match excludes far more than users typically expect,
            // so bare negated atoms match substrings, like fzf and nucleo
            _ if negated => Some(Matching::Substring),
            _ => None,
        };

        Self {
            pattern: atom.to_string(),
            negated,
            needle,
            config: PatternConfig::default().matching(matching),
        }
    }

    /// Parses a query of whitespace separated atoms (see [`Pattern::parse`]),
    /// e.g. `foo !^bar` matches haystacks that fuzzy match `foo` and don't
    /// start with `bar`. Escape a literal space with a backslash, e.g.
    /// `foo\ bar` is a single atom. Atoms with an empty needle, e.g. `!` or
    /// `^$`, are dropped.
    ///
    /// The returned patterns carry only the [`Matching`] mode derived from the
    /// syntax. Any other per-pattern override is left as `None` and
    /// inherits the matcher's [`Config`]. Set other [`PatternConfig`]
    /// fields on the results to override per-pattern. For example, setting
    /// the max typos based on needle length:
    ///
    /// ```
    /// use neo_frizbee::{Config, Matcher, Pattern};
    ///
    /// let patterns = Pattern::parse_query("foo longerneedle")
    ///     .into_iter()
    ///     .map(|pattern| {
    ///         let max_typos = (pattern.needle.len() / 4) as u16;
    ///         pattern.max_typos(Some(max_typos))
    ///     })
    ///     .collect::<Vec<_>>();
    ///
    /// let mut matcher = Matcher::from_patterns(&patterns, &Config::default());
    /// ```
    pub fn parse_query(query: &str) -> Vec<Pattern> {
        let mut patterns = vec![];
        let mut start: Option<usize> = None;
        let mut escaped = false;

        let mut push = |atom: &str| {
            let pattern = Self::parse(atom);
            if !pattern.needle.is_empty() {
                patterns.push(pattern);
            }
        };

        for (i, c) in query.char_indices() {
            if escaped {
                // Previous char was an escaping backslash; keep this char in the atom
                escaped = false;
            } else if c == '\\' {
                start.get_or_insert(i);
                escaped = true;
            } else if c.is_whitespace() {
                if let Some(s) = start.take() {
                    push(&query[s..i]);
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            push(&query[s..]);
        }

        patterns
    }
}

/// Per-pattern overrides for the matcher's [`Config`]. Every field is optional
/// and falls back to the matcher's [`Config`] when left as `None` (see
/// [`PatternConfig::resolve`])
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct PatternConfig {
    /// Per-pattern override for [`crate::Config::max_typos`]; `None` inherits
    /// it.
    ///
    /// Because [`crate::Config::max_typos`] is itself `Option`, there is no way
    /// to request unlimited typos for a single pattern while the matcher's
    /// config sets a limit.
    pub max_typos: Option<u16>,
    /// Per-pattern override for [`crate::Config::casing`]; `None` inherits it.
    pub casing: Option<CaseMatching>,
    /// Per-pattern override for [`crate::Config::unicode`]; `None` inherits it.
    pub unicode: Option<UnicodeMatching>,
    /// Per-pattern override for [`crate::Config::matching`]; `None` inherits
    /// it.
    pub matching: Option<Matching>,
    /// Per-pattern override for [`crate::Config::scoring`]; `None` inherits it.
    pub scoring: Option<Scoring>,
}

impl PatternConfig {
    /// Resolves this pattern's overrides against the matcher's [`Config`],
    /// using the matcher's value for any field left as `None`. The returned
    /// config's `sort` is always the matcher's, as result ordering isn't a
    /// per-pattern concern.
    pub fn resolve(&self, config: &Config) -> Config {
        Config {
            max_typos: self.max_typos.or(config.max_typos),
            casing: self.casing.unwrap_or(config.casing),
            unicode: self.unicode.unwrap_or(config.unicode),
            matching: self.matching.unwrap_or(config.matching),
            scoring: self
                .scoring
                .clone()
                .unwrap_or_else(|| config.scoring.clone()),
            sort: config.sort,
        }
    }

    /// Sets the matching mode
    pub fn matching(mut self, matching: Option<Matching>) -> Self {
        self.matching = matching;
        self
    }

    /// Sets the maximum number of typos allowed
    pub fn max_typos(mut self, max_typos: Option<u16>) -> Self {
        self.max_typos = max_typos;
        self
    }

    /// Sets the casing mode
    pub fn casing(mut self, casing: Option<CaseMatching>) -> Self {
        self.casing = casing;
        self
    }

    /// Sets the unicode mode
    pub fn unicode(mut self, unicode: Option<UnicodeMatching>) -> Self {
        self.unicode = unicode;
        self
    }

    /// Sets the scoring
    pub fn scoring(mut self, scoring: Option<Scoring>) -> Self {
        self.scoring = scoring;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_parse(atom: &str, needle: &str, matching: Option<Matching>, negated: bool) {
        let pattern = Pattern::parse(atom);
        assert_eq!(pattern.pattern, atom);
        assert_eq!(pattern.needle, needle, "atom: {atom:?}");
        assert_eq!(pattern.config.matching, matching, "atom: {atom:?}");
        assert_eq!(pattern.negated, negated, "atom: {atom:?}");
    }

    #[test]
    fn parse_selects_matching_mode() {
        assert_parse("foo", "foo", None, false);
        assert_parse("^foo", "foo", Some(Matching::Prefix), false);
        assert_parse("foo$", "foo", Some(Matching::Suffix), false);
        assert_parse("'foo", "foo", Some(Matching::Substring), false);
        assert_parse("^foo$", "foo", Some(Matching::Exact), false);
    }

    #[test]
    fn parse_negation() {
        // Bare negated atoms match substrings, like fzf and nucleo
        assert_parse("!foo", "foo", Some(Matching::Substring), true);
        assert_parse("!^foo", "foo", Some(Matching::Prefix), true);
        assert_parse("!foo$", "foo", Some(Matching::Suffix), true);
        assert_parse("!'foo", "foo", Some(Matching::Substring), true);
        assert_parse("!^foo$", "foo", Some(Matching::Exact), true);
    }

    #[test]
    fn parse_escapes_special_syntax() {
        assert_parse("\\^foo", "^foo", None, false);
        assert_parse("foo\\$", "foo$", None, false);
        assert_parse("\\'foo", "'foo", None, false);
        assert_parse("\\!foo", "!foo", None, false);
        assert_parse("foo\\ bar", "foo bar", None, false);
        assert_parse("!\\^foo", "^foo", Some(Matching::Substring), true);
        assert_parse("!\\!foo", "!foo", Some(Matching::Substring), true);
    }

    #[test]
    fn parse_escaped_backslash_before_operator() {
        assert_parse("foo\\\\$", "foo\\\\", Some(Matching::Suffix), false);
        // Backslashes before non-special characters are literal
        assert_parse("foo\\bar", "foo\\bar", None, false);
        assert_parse("foo\\", "foo\\", None, false);
        // The first two backslashes pair up, the third escapes the space
        assert_parse("a\\\\\\ b", "a\\\\ b", None, false);
    }

    #[test]
    fn parse_query_splits_atoms() {
        let patterns = Pattern::parse_query("foo !^bar");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0], Pattern::parse("foo"));
        assert_eq!(patterns[1], Pattern::parse("!^bar"));

        let patterns = Pattern::parse_query("  foo \t bar  ");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].needle, "foo");
        assert_eq!(patterns[1].needle, "bar");
    }

    #[test]
    fn parse_query_escaped_space() {
        let patterns = Pattern::parse_query("foo\\ bar baz");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].needle, "foo bar");
        assert_eq!(patterns[1].needle, "baz");
    }

    #[test]
    fn parse_query_escaped_backslash_before_space_splits() {
        // The escaped backslash consumes the escape, so the space still separates atoms
        let patterns = Pattern::parse_query("foo\\\\ bar");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].needle, "foo\\\\");
        assert_eq!(patterns[1].needle, "bar");
    }

    #[test]
    fn parse_query_drops_empty_atoms() {
        assert!(Pattern::parse_query("").is_empty());
        assert!(Pattern::parse_query("   ").is_empty());
        assert!(Pattern::parse_query("! ^$ '").is_empty());
    }
}
