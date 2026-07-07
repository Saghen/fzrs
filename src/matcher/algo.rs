use crate::prefilter::{Kernel as PrefilterKernel, Window};
use crate::smith_waterman::Kernel as SmithWatermanKernel;
use crate::{Config, Match, MatchIndices};
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// Magic numbers for `TYPOS` specialization keys beyond literal 0/1/2
pub(super) const MANY_TYPOS: u16 = u16::MAX;
pub(super) const NO_PREFILTER: u16 = u16::MAX - 1;

/// Fully inlined per-backend implementations, specialized for each
/// configuration (0 typos, 1 typo, unicode variants, ...) and built with the
/// backend's `#[target_feature]`.
///
/// # Safety
/// The backend's required CPU features must be available
pub(crate) trait Specialized: Sized {
    unsafe fn build(needle: &str, config: &Config) -> Self;

    unsafe fn match_list<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    );

    unsafe fn match_list_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
    ) -> Vec<MatchIndices>;

    unsafe fn match_one<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<Match>;

    unsafe fn match_one_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<MatchIndices>;
}

#[derive(Debug, Clone)]
pub struct MatcherImpl<P: PrefilterKernel, S: SmithWatermanKernel> {
    needle: String,
    config: Config,
    min_haystack_len: usize,
    prefilter: P,
    smith_waterman: S,
}

impl<P, S> MatcherImpl<P, S>
where
    P: PrefilterKernel,
    S: SmithWatermanKernel,
{
    #[inline(always)]
    pub fn new(needle: &str, config: &Config) -> Self {
        let case_sensitive = config.casing.respects_case_for(needle);
        Self {
            needle: needle.to_string(),
            config: *config,
            min_haystack_len: config
                .max_typos
                .map(|max| needle.chars().count().saturating_sub(max as usize))
                .unwrap_or(0),
            prefilter: P::new(needle, case_sensitive),
            smith_waterman: S::new(needle, &config.scoring, case_sensitive),
        }
    }

    pub fn is_available() -> bool {
        P::is_available() && S::is_available()
    }

    #[inline(always)]
    pub(super) fn match_list_into_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        let max_typos = self.max_typos_runtime::<TYPOS>();
        for (index, haystack_str) in (haystack_index_offset..).zip(haystacks.iter()) {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= self.min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let (trimmed, start_pos, include_exact) =
                        trim_haystack(haystack, start_pos, end_pos);
                    matches.push(self.smith_waterman_one::<UNICODE>(
                        trimmed,
                        index,
                        start_pos,
                        include_exact,
                    ));
                }
            }
        }
    }

    /// Single-haystack path for `Matcher::match_iter`, which branches on the
    /// typo/unicode configuration at runtime rather than expanding a hot loop
    /// per configuration, so it monomorphizes once per backend. Each kernel
    /// call crosses the `#[target_feature]` boundary instead of inlining into
    /// a shared loop.
    #[inline(always)]
    pub(super) fn match_one_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<Match> {
        let haystack = haystack.as_ref().as_bytes();
        let max_typos = self.max_typos_runtime::<TYPOS>();
        let original_len = haystack.len();
        if original_len < self.min_haystack_len {
            return None;
        }

        let (matched, start_pos, end_pos) =
            self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
        if !matched {
            return None;
        }

        let (trimmed, start_pos, include_exact) = trim_haystack(haystack, start_pos, end_pos);
        Some(self.smith_waterman_one::<UNICODE>(trimmed, index, start_pos, include_exact))
    }

    /// Single-haystack path for `Matcher::match_iter_indices`, mirroring
    /// `match_one_impl` but returning the matched character indices. Like the
    /// list variant it branches on the typo/unicode configuration at runtime so
    /// it monomorphizes once per backend.
    #[inline(always)]
    pub(super) fn match_one_indices_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<MatchIndices> {
        let haystack = haystack.as_ref().as_bytes();
        let max_typos = self.max_typos_runtime::<TYPOS>();
        let max_typos_opt = if TYPOS == NO_PREFILTER {
            None
        } else {
            Some(max_typos)
        };
        let original_len = haystack.len();
        if original_len < self.min_haystack_len {
            return None;
        }

        let (matched, start_pos, end_pos) =
            self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
        if !matched {
            return None;
        }

        let (trimmed, start_pos, include_exact) = trim_haystack(haystack, start_pos, end_pos);
        Some(self.smith_waterman_indices_one::<UNICODE>(
            trimmed,
            start_pos,
            index,
            include_exact,
            max_typos_opt,
        ))
    }

    #[inline(always)]
    fn prefilter_haystack<const TYPOS: u16, const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> Window {
        match TYPOS {
            NO_PREFILTER => (true, 0, haystack.len()),
            0 if UNICODE => self.prefilter.match_haystack_unicode(haystack),
            0 => self.prefilter.match_haystack(haystack),
            1 if UNICODE => self.prefilter.match_haystack_unicode_1_typo(haystack),
            1 => self.prefilter.match_haystack_1_typo(haystack),
            2 if UNICODE => self.prefilter.match_haystack_unicode_2_typos(haystack),
            2 => self.prefilter.match_haystack_2_typos(haystack),
            MANY_TYPOS if UNICODE => self
                .prefilter
                .match_haystack_unicode_many_typos(haystack, max_typos),
            MANY_TYPOS => self
                .prefilter
                .match_haystack_many_typos(haystack, max_typos),
            _ => unreachable!("unsupported typo count specialization"),
        }
    }

    #[inline(always)]
    pub(super) fn match_list_indices_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
    ) -> Vec<MatchIndices> {
        let max_typos = self.max_typos_runtime::<TYPOS>();
        let max_typos_opt = if TYPOS == NO_PREFILTER {
            None
        } else {
            Some(max_typos)
        };
        let mut matches = vec![];
        for (index, haystack_str) in haystacks.iter().enumerate() {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= self.min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let (trimmed, start_pos, include_exact) =
                        trim_haystack(haystack, start_pos, end_pos);
                    matches.push(self.smith_waterman_indices_one::<UNICODE>(
                        trimmed,
                        start_pos,
                        index as u32,
                        include_exact,
                        max_typos_opt,
                    ));
                }
            }
        }
        matches
    }

    #[inline(always)]
    fn smith_waterman_one<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        index: u32,
        haystack_start_pos: usize,
        include_exact: bool,
    ) -> Match {
        let mut score = if UNICODE {
            self.smith_waterman
                .score_haystack_unicode(haystack, haystack_start_pos)
        } else {
            self.smith_waterman
                .score_haystack(haystack, haystack_start_pos)
        };

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score = score.saturating_add(self.config.scoring.exact_match_bonus);
        }

        #[cfg(not(feature = "match_end_col"))]
        let _ = haystack_start_pos;

        Match {
            index,
            score,
            exact,
            #[cfg(feature = "match_end_col")]
            end_col: self
                .smith_waterman
                .match_end_col(haystack, UNICODE)
                .saturating_add(haystack_start_pos.min(u16::MAX as usize) as u16),
        }
    }

    #[inline(always)]
    fn smith_waterman_indices_one<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
        index: u32,
        include_exact: bool,
        max_typos: Option<u16>,
    ) -> MatchIndices {
        let (mut score, indices) = if UNICODE {
            self.smith_waterman.score_haystack_unicode_indices(
                haystack,
                haystack_start_pos,
                max_typos,
            )
        } else {
            self.smith_waterman
                .score_haystack_indices(haystack, haystack_start_pos, max_typos)
        };

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score = score.saturating_add(self.config.scoring.exact_match_bonus);
        }

        MatchIndices {
            index,
            score,
            exact,
            indices,
        }
    }

    /// The runtime typo budget for a `TYPOS` specialization: 0/1/2 encode the
    /// count directly, `MANY_TYPOS` reads it from the config, and
    /// `NO_PREFILTER` never uses it.
    #[inline(always)]
    fn max_typos_runtime<const TYPOS: u16>(&self) -> u16 {
        if TYPOS == MANY_TYPOS {
            self.config.max_typos.unwrap_or(0)
        } else {
            TYPOS
        }
    }
}

/// Trims the haystack to the prefilter's window, returning the trimmed slice,
/// the trimmed start position, and whether the window covers the full haystack
/// (making it eligible for the exact match bonus)
#[inline(always)]
fn trim_haystack(haystack: &[u8], start_pos: usize, end_pos: usize) -> (&[u8], usize, bool) {
    // substract 1 so that we add the delimiter bonus from the first char
    // otherwise, we would never see it in the smith waterman
    let start_pos = start_pos.saturating_sub(1);
    let include_exact = start_pos == 0 && end_pos == haystack.len();
    (&haystack[start_pos..end_pos], start_pos, include_exact)
}

#[cfg(test)]
mod tests {
    use crate::{Config, Matcher, Scoring, SortStrategy};

    #[test]
    fn all_zero_scoring_does_not_divide_by_zero() {
        let config = Config::default().scoring(Scoring {
            match_score: 0,
            mismatch_penalty: 0,
            gap_open_penalty: 0,
            gap_extend_penalty: 0,
            prefix_bonus: 0,
            capitalization_bonus: 0,
            matching_case_bonus: 0,
            exact_match_bonus: 0,
            delimiter_bonus: 0,
        });
        Matcher::new("foo", &config).match_list(&["foobar"]);
    }

    #[test]
    fn gap_open_below_gap_extend_does_not_underflow() {
        let config = Config::default().scoring(Scoring {
            gap_open_penalty: 1,
            gap_extend_penalty: 5,
            ..Scoring::default()
        });
        Matcher::new("foo", &config).match_list(&["foobar", "fabco"]);
    }

    #[test]
    fn huge_bonuses_saturate_instead_of_panicking() {
        let config = Config::default().scoring(Scoring {
            capitalization_bonus: 60000,
            matching_case_bonus: 40000,
            ..Scoring::default()
        });
        let matches = Matcher::new("f", &config).match_list(&["f", "z"]);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn unicode_needles_score_per_char_row() {
        // the unicode path uses one score row per char (8), not per byte (24)
        let needle = "一二三四五六七八";
        let config = Config::default().scoring(Scoring {
            capitalization_bonus: 4000,
            ..Scoring::default()
        });
        let matches = Matcher::new(needle, &config).match_list(&[needle]);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn greedy_fallback_membership_agrees_between_match_list_and_indices() {
        // the match window exceeds MAX_HAYSTACK_LEN so the greedy fallback runs, and it
        // can't find 'c': both APIs must still return the prefiltered haystack, with
        // score 0 and no indices
        let haystack = format!("a{}b", "z".repeat(1100));
        let config = Config::default().max_typos(Some(1));
        let matches = Matcher::new("abc", &config).match_list(&[haystack.as_str()]);
        let indices = Matcher::new("abc", &config).match_list_indices(&[haystack.as_str()]);
        assert_eq!(matches.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(matches[0].score, indices[0].score);
        assert!(indices[0].indices.is_empty());
    }

    #[test]
    fn penalty_above_u8_range_is_not_truncated() {
        // mismatch_penalty 260 previously wrapped to 4 in the u8 backend's splat,
        // so raising the penalty raised the score
        let score = |mismatch_penalty| {
            let config = Config::default().max_typos(Some(1)).scoring(Scoring {
                mismatch_penalty,
                ..Scoring::default()
            });
            Matcher::new("abc", &config).match_list(&["aXc"])[0].score
        };
        assert!(score(260) <= score(255));
    }

    #[test]
    fn zero_gap_capitalization_scores_do_not_saturate_u8() {
        // with no gap penalties, every matched char can earn the full capitalization
        // bonus, so the u8 backend's 255 ceiling must not be selected
        let config = Config::default().scoring(Scoring {
            match_score: 40,
            capitalization_bonus: 40,
            mismatch_penalty: 0,
            gap_open_penalty: 0,
            gap_extend_penalty: 0,
            prefix_bonus: 0,
            matching_case_bonus: 0,
            exact_match_bonus: 0,
            delimiter_bonus: 0,
        });
        let matches = Matcher::new("BBBB", &config).match_list(&["aBaBaBaB"]);
        assert_eq!(matches[0].score, 4 * (40 + 40));
    }

    #[test]
    fn unsorted_output_preserves_candidate_order() {
        let haystacks = ["foo", "nomatch", "xfoo", "f_o_o", "bar"];
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("foo", &config).match_list(&haystacks);
        assert_eq!(
            matches
                .iter()
                .map(|match_| match_.index)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn match_list_indices_reports_expected_public_indices() {
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
    fn filtered_match_end_col_uses_original_haystack_offsets() {
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("abc", &config).match_list(&["xxabcxx"]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].end_col, 4);
    }
}
