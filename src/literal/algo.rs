use super::rank::rare_byte_offsets;
use crate::prefilter::algo::load_window;
use crate::prefilter::backend::{Backend, BitMaskOps};
use crate::prefilter::{UnicodeChar, case_needle, case_needle_unicode};
use crate::{Config, Match, MatchIndices, Matching, Scoring};
use alloc::{vec, vec::Vec};

/// Literal matching: exact / prefix / suffix / substring
/// Specialized for one SIMD [`crate::prefilter::backend::Backend`] supporting
/// both ASCII and Unicode Identical scoring to Smith-Waterman
#[derive(Debug, Clone)]
pub(crate) struct LiteralImpl<B: Backend> {
    mode: Matching,
    scoring: Scoring,
    needle_len: usize,
    /// Per-byte `(original, opposite-case)` bytes for ASCII case-insensitive
    /// matching
    needle_ascii: Vec<(u8, u8)>,
    /// Per-codepoint needle, read only on the unicode path. Holds each
    /// character's UTF-8 bytes and its opposite-case bytes, used for
    /// whole-codepoint case-insensitive matching
    needle_unicode: Vec<UnicodeChar>,
    /// Needle byte offsets of the two rarest bytes, `seed_a_off <= seed_b_off`,
    /// chosen as the two rarest bytes (see [`rare_byte_offsets`])
    seed_a_off: usize,
    seed_b_off: usize,
    /// Splatted `(original, opposite-case)` bytes at `seed_a_off` /
    /// `seed_b_off`, the two scan seeds
    seed_a: (B::Chunk, B::Chunk),
    seed_b: (B::Chunk, B::Chunk),
}

impl<B: Backend> LiteralImpl<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub(crate) unsafe fn new(needle: &str, config: &Config) -> Self {
        let case_sensitive = config.casing.respects_case_for(needle);
        let unicode = config.unicode.respects_unicode_for(needle);
        let needle_ascii = case_needle(needle.as_bytes(), case_sensitive);
        let needle_unicode = case_needle_unicode(needle, case_sensitive);

        // Per-byte `(original, opposite-case)` pairs used to splat the prefilter seeds.
        // On the unicode path this flattens each codepoint's bytes; on the
        // ASCII path it is `needle_ascii`.
        let seed_bytes: Vec<(u8, u8)> = if unicode {
            needle_unicode
                .iter()
                .flat_map(|c| (0..c.len).map(|i| (c.chars[i], c.flipped_chars[i])))
                .collect()
        } else {
            needle_ascii.clone()
        };

        // Seed the two-byte prefilter on the two rarest needle bytes. A single-byte (or
        // empty) needle has no pair, so both seeds collapse onto offset 0.
        let (seed_a_off, seed_b_off) = if seed_bytes.len() >= 2 {
            rare_byte_offsets(needle.as_bytes())
        } else {
            (0, 0)
        };
        let splat = |off: usize| {
            // Falls back to (0, 0) only for an empty needle, which never reaches the match
            // methods
            let (orig, flipped) = seed_bytes.get(off).copied().unwrap_or_default();
            unsafe { (B::splat(orig), B::splat(flipped)) }
        };
        let seed_a = splat(seed_a_off);
        let seed_b = splat(seed_b_off);

        Self {
            mode: config.matching,
            scoring: config.scoring,
            needle_len: needle.len(),
            needle_ascii,
            needle_unicode,
            seed_a_off,
            seed_b_off,
            seed_a,
            seed_b,
        }
    }

    pub(crate) fn is_available() -> bool {
        B::is_available()
    }

    #[inline(always)]
    pub(super) unsafe fn match_list_impl<const UNICODE: bool, H: AsRef<str>>(
        &self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        for (index, haystack) in (haystack_index_offset..).zip(haystacks.iter()) {
            if let Some(m) = unsafe { self.match_one_impl::<UNICODE, &H>(haystack, index) } {
                matches.push(m);
            }
        }
    }

    #[inline(always)]
    pub(super) unsafe fn match_one_impl<const UNICODE: bool, H: AsRef<str>>(
        &self,
        haystack: H,
        index: u32,
    ) -> Option<Match> {
        let haystack = haystack.as_ref().as_bytes();
        let (pos, score) = unsafe { self.find::<UNICODE>(haystack) }?;
        let exact = pos == 0 && self.needle_len == haystack.len();
        Some(Match {
            index,
            score,
            exact,
            #[cfg(feature = "match_end_col")]
            end_col: (pos + self.needle_len)
                .saturating_sub(1)
                .min(u16::MAX as usize) as u16,
        })
    }

    #[inline(always)]
    pub(super) unsafe fn match_list_indices_impl<const UNICODE: bool, H: AsRef<str>>(
        &self,
        haystacks: &[H],
    ) -> Vec<MatchIndices> {
        let mut matches = vec![];
        for (index, haystack) in haystacks.iter().enumerate() {
            if let Some(m) =
                unsafe { self.match_one_indices_impl::<UNICODE, &H>(haystack, index as u32) }
            {
                matches.push(m);
            }
        }
        matches
    }

    #[inline(always)]
    pub(super) unsafe fn match_one_indices_impl<const UNICODE: bool, H: AsRef<str>>(
        &self,
        haystack: H,
        index: u32,
    ) -> Option<MatchIndices> {
        let haystack = haystack.as_ref().as_bytes();
        let (pos, score) = unsafe { self.find::<UNICODE>(haystack) }?;
        let exact = pos == 0 && self.needle_len == haystack.len();
        // Every byte of the matched run is a matched index, but add in reverse order to
        // match the fuzzy matcher implementation
        let indices = (pos..pos + self.needle_len)
            .rev()
            .map(|i| i as u32)
            .collect();
        Some(MatchIndices {
            index,
            score,
            exact,
            indices,
        })
    }

    /// Verifies that the needle matches the haystack starting at byte `pos`
    #[inline(always)]
    fn matches_at<const UNICODE: bool>(&self, haystack: &[u8], pos: usize) -> bool {
        if UNICODE {
            let mut k = pos;
            for c in &self.needle_unicode {
                let bytes = &haystack[k..k + c.len];
                if bytes != &c.chars[..c.len] && bytes != &c.flipped_chars[..c.len] {
                    return false;
                }
                k += c.len;
            }
        } else {
            for (k, &(orig, flipped)) in self.needle_ascii.iter().enumerate() {
                let byte = haystack[pos + k];
                if byte != orig && byte != flipped {
                    return false;
                }
            }
        }
        true
    }

    /// Score contribution of a single matched scalar whose start byte is at
    /// haystack index `start`. `matched_exact_case` is true when the
    /// haystack scalar equals the needle's original case.
    #[inline(always)]
    fn score_scalar(&self, haystack: &[u8], start: usize, matched_exact_case: bool) -> u16 {
        let s = &self.scoring;
        let mut score = s.match_score;
        if matched_exact_case {
            score = score.saturating_add(s.matching_case_bonus);
        }
        if start == 0 {
            score = score.saturating_add(s.prefix_bonus);
        } else {
            let byte = haystack[start];
            let prev = haystack[start - 1];
            if byte.is_ascii_uppercase() && prev.is_ascii_lowercase() {
                score = score.saturating_add(s.capitalization_bonus);
            }
            if is_delimiter(prev) && !is_delimiter(byte) {
                score = score.saturating_add(s.delimiter_bonus);
            }
        }
        score
    }

    /// Scores a contiguous match at byte `pos`, summing one
    /// [`Self::score_scalar`] per needle scalar: a byte on the ASCII path
    /// or a codepoint on the unicode path
    #[inline(always)]
    fn score_at<const UNICODE: bool>(&self, haystack: &[u8], pos: usize) -> u16 {
        let mut score = 0u16;
        if UNICODE {
            let mut start = pos;
            for c in &self.needle_unicode {
                let matched_exact_case = haystack[start..start + c.len] == c.chars[..c.len];
                score =
                    score.saturating_add(self.score_scalar(haystack, start, matched_exact_case));
                start += c.len;
            }
        } else {
            for (k, &(orig, _)) in self.needle_ascii.iter().enumerate() {
                let start = pos + k;
                score = score.saturating_add(self.score_scalar(
                    haystack,
                    start,
                    haystack[start] == orig,
                ));
            }
        }

        if pos == 0 && self.needle_len == haystack.len() {
            score = score.saturating_add(self.scoring.exact_match_bonus);
        }
        score
    }

    /// Returns the matched byte position (start) if the haystack matches under
    /// the configured mode as well as the score.
    /// For substring, it checks all positions to find the best-score,
    /// preferring earlier matches when tied.
    #[inline(always)]
    unsafe fn find<const UNICODE: bool>(&self, haystack: &[u8]) -> Option<(usize, u16)> {
        let needle_len = self.needle_len;
        if haystack.len() < needle_len {
            return None;
        }

        match self.mode {
            Matching::Fuzzy => unreachable!("fuzzy matching does not use the literal backend"),
            Matching::Exact => (haystack.len() == needle_len
                && self.matches_at::<UNICODE>(haystack, 0))
            .then(|| (0, self.score_at::<UNICODE>(haystack, 0))),
            Matching::Prefix => self
                .matches_at::<UNICODE>(haystack, 0)
                .then(|| (0, self.score_at::<UNICODE>(haystack, 0))),
            Matching::Suffix => {
                let pos = haystack.len() - needle_len;
                self.matches_at::<UNICODE>(haystack, pos)
                    .then(|| (pos, self.score_at::<UNICODE>(haystack, pos)))
            }
            Matching::Substring => unsafe { self.find_substring::<UNICODE>(haystack) },
        }
    }

    /// Two-byte SIMD prefilter (similar to `memchr::memmem`)
    ///
    /// Scan the string for both seed bytes (the two rarest bytes from the
    /// needle), and on a match, perform a scalar scan for the rest of the
    /// needle.
    #[inline(always)]
    unsafe fn find_substring<const UNICODE: bool>(&self, haystack: &[u8]) -> Option<(usize, u16)> {
        let len = haystack.len();
        let needle_len = self.needle_len;
        debug_assert!(needle_len > 0, "empty needles are handled by the caller");
        let last_start = len - needle_len + 1;

        let mut best: Option<(usize, u16)> = None;

        let mut start = 0usize;
        while start < last_start {
            // Mask out any over-read lanes and any lanes that can't possibly match the
            // needle
            let usable = last_start - start;
            let valid = if usable >= B::LANES {
                B::Mask::all()
            } else {
                B::Mask::first_n(usable)
            };

            // Seed window lane `k` corresponds to position `pos = start + k`: it reads
            // `haystack[start + seed_off + k] = haystack[pos + seed_off]`.
            let (chunk_a, _) = unsafe { load_window::<B>(haystack, start + self.seed_a_off, len) };
            let hits_a = unsafe { B::occ(chunk_a, self.seed_a) }.and(valid);
            // Skip the second load entirely when the first seed matches nowhere in this
            // window.
            if hits_a.is_zero() {
                start += B::LANES;
                continue;
            }

            // A single-byte needle has only one seed (both offsets will be 0), so skip
            // Branch predictor will predict this perfectly, so there's no performance cost
            let mut hits = if self.seed_a_off != self.seed_b_off {
                let (chunk_b, _) =
                    unsafe { load_window::<B>(haystack, start + self.seed_b_off, len) };
                hits_a.and(unsafe { B::occ(chunk_b, self.seed_b) })
            } else {
                hits_a
            };
            while !hits.is_zero() {
                let pos = start + unsafe { B::first_hit_pos(hits) };
                hits = hits.clear_through_lowest(hits);
                // We've verified the seeds but we have to check the rest of the needle matches
                // now
                if (!UNICODE && needle_len <= 2) || self.matches_at::<UNICODE>(haystack, pos) {
                    let score = self.score_at::<UNICODE>(haystack, pos);
                    if best.is_none_or(|(_, best_score)| score > best_score) {
                        best = Some((pos, score));
                    }
                }
            }
            start += B::LANES;
        }
        best
    }
}

#[inline(always)]
fn is_delimiter(byte: u8) -> bool {
    byte <= 127 && !byte.is_ascii_alphanumeric()
}
