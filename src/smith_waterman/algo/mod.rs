use crate::smith_waterman::Kernel;
use crate::{
    Scoring,
    prefilter::{case_needle, case_needle_unicode},
    smith_waterman::greedy::match_greedy,
};
use alloc::{string::ToString, vec, vec::Vec};

use super::SmithWaterman;
use super::alignment_iter::Alignment;
use super::backend::{Backend, BytesVec, ScoreVec};
use super::matrix::Matrix;

mod ascii;
pub(crate) mod ascii_gap;
mod unicode;
pub(crate) mod unicode_gap;

pub(crate) const MAX_HAYSTACK_LEN: usize = 1024;

impl<B: Backend> Kernel for SmithWaterman<B> {
    fn new(needle: &str, scoring: &Scoring, case_sensitive: bool) -> Self {
        let needle_simd = case_needle(needle.as_bytes(), case_sensitive)
            .iter()
            .map(|(c1, c2)| unsafe { (B::Bytes::splat(*c1), B::Bytes::splat(*c2)) })
            .collect();
        let needle_unicode = case_needle_unicode(needle, case_sensitive);
        let needle_len = needle.len();
        let unicode_pending_gap_open_masks = (0..(needle_unicode.len() + 1))
            .map(|_| unsafe { B::Score::zero() })
            .collect();
        Self {
            needle: needle.to_string(),
            needle_simd,
            needle_unicode,
            case_sensitive,
            scoring: *scoring,
            score_matrix: Matrix::new(needle_len, MAX_HAYSTACK_LEN),
            match_masks: Matrix::new(needle_len, MAX_HAYSTACK_LEN),
            unicode_pending_gap_open_masks,
            haystack_chunks: 0,
        }
    }

    fn is_available() -> bool {
        B::is_available()
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    fn score_haystack_indices(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
        max_typos: Option<u16>,
    ) -> (u16, Vec<u32>) {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(
                self.needle.as_bytes(),
                haystack,
                &self.scoring,
                self.case_sensitive,
                haystack_start_pos == 0,
            )
            .map(|(score, mut indices)| {
                for index in &mut indices {
                    *index = u32::try_from(*index as usize + haystack_start_pos)
                        .expect("haystack byte index will overflow u32");
                }
                indices.reverse();
                (score, indices)
            })
            .unwrap_or_else(|| (0, vec![]));
        }

        let score = self.score_haystack(haystack, haystack_start_pos == 0);
        if score == 0 {
            return (score, Vec::new());
        }

        let needle_len = self.needle.len();
        let mut indices = Vec::with_capacity(needle_len);
        for pos in self.iter_alignment_path(needle_len, haystack_start_pos, None, score, max_typos)
        {
            match pos {
                Some(Alignment::Match((_, haystack_idx))) => indices.push(haystack_idx as u32),
                Some(_) => {}
                // TODO: it's possible for us to lose alignment due to score == 0
                // but to stay consistent with results of `match_list`, we simply
                // don't return the full list of indices
                None => break,
            }
        }

        (score, indices)
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    fn score_haystack_unicode_indices(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
        max_typos: Option<u16>,
    ) -> (u16, Vec<u32>) {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(
                self.needle.as_bytes(),
                haystack,
                &self.scoring,
                self.case_sensitive,
                haystack_start_pos == 0,
            )
            .map(|(score, mut indices)| {
                for index in &mut indices {
                    *index = u32::try_from(*index as usize + haystack_start_pos)
                        .expect("haystack byte index will overflow u32");
                }
                indices.reverse();
                (score, indices)
            })
            .unwrap_or_else(|| (0, vec![]));
        }

        let score = self.score_haystack_unicode(haystack, haystack_start_pos == 0);
        if score == 0 {
            return (score, Vec::new());
        }

        let mut indices = Vec::with_capacity(self.needle.len());
        let mut prev_haystack_idx = usize::MAX;
        for pos in self.iter_alignment_path(
            self.needle_unicode.len(),
            haystack_start_pos,
            Some(haystack),
            score,
            max_typos,
        ) {
            match pos {
                Some(Alignment::Match((needle_idx, haystack_idx))) => {
                    if prev_haystack_idx != haystack_idx {
                        let len = self.needle_unicode[needle_idx].len;
                        indices.extend((0..len).rev().map(|offset| (haystack_idx + offset) as u32));
                        prev_haystack_idx = haystack_idx;
                    }
                }
                Some(_) => {}
                // TODO: it's possible for us to lose alignment due to score == 0
                // but to stay consistent with results of `match_list`, we simply
                // don't return the full list of indices
                None => break,
            }
        }

        (score, indices)
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    fn score_haystack(&mut self, haystack: &[u8], haystack_start_pos: usize) -> u16 {
        SmithWaterman::score_haystack(self, haystack, haystack_start_pos == 0)
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    fn score_haystack_unicode(&mut self, haystack: &[u8], haystack_start_pos: usize) -> u16 {
        SmithWaterman::score_haystack_unicode(self, haystack, haystack_start_pos == 0)
    }

    #[cfg(feature = "match_end_col")]
    fn match_end_col(&self, haystack: &[u8], unicode: bool) -> u16 {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(
                self.needle.as_bytes(),
                haystack,
                &self.scoring,
                self.case_sensitive,
                true,
            )
            .and_then(|(_, indices)| indices.last().copied())
            .unwrap_or(0)
            .min(u16::MAX as u32) as u16;
        }

        // The unicode scorer walks one row per codepoint, the ASCII scorer one per byte
        let final_row = if unicode {
            self.needle_unicode.len()
        } else {
            self.needle.len()
        };
        let mut match_end_col: u16 = 0;
        let mut max_score = 0;
        for col_idx in 1..(haystack.len().div_ceil(B::LANES) + 1) {
            let chunk_scores = self.score_matrix.get(final_row, col_idx);
            let chunk_max_score = unsafe { chunk_scores.horizontal_max() };
            if chunk_max_score > max_score {
                max_score = chunk_max_score;
                let lane = unsafe { chunk_scores.find_lane(chunk_max_score) };
                match_end_col = ((col_idx - 1) * B::LANES + lane) as u16;
            }
        }
        match_end_col
    }
}
