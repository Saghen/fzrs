//! Definitions for all target feature-specific implementations of the matcher

use super::algo::{MatcherImpl, Specialized};
use crate::{Config, Match, MatchIndices};
use alloc::vec::Vec;

#[cfg(target_arch = "aarch64")]
use crate::literal::LiteralNEON;
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
use crate::literal::LiteralScalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::literal::LiteralWasm;
#[cfg(target_arch = "x86_64")]
use crate::literal::{LiteralAVX, LiteralAVX512, LiteralSSE};

#[cfg(target_arch = "aarch64")]
use crate::prefilter::backend::PrefilterNEON;
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
use crate::prefilter::backend::PrefilterScalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::prefilter::backend::PrefilterWasm;
#[cfg(target_arch = "x86_64")]
use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};
#[cfg(target_arch = "x86_64")]
use crate::smith_waterman::{
    SmithWatermanAVX, SmithWatermanAVX512, SmithWatermanAVX512U8, SmithWatermanAVXU8,
    SmithWatermanSSE, SmithWatermanSSEU8,
};
#[cfg(target_arch = "aarch64")]
use crate::smith_waterman::{SmithWatermanNEON, SmithWatermanNEONU8};
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
use crate::smith_waterman::{SmithWatermanScalar, SmithWatermanScalarU8};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::smith_waterman::{SmithWatermanWasm, SmithWatermanWasmU8};

#[cfg(target_arch = "x86_64")]
pub type MatcherAVX512U8 = MatcherImpl<PrefilterAVX512, SmithWatermanAVX512U8>;
#[cfg(target_arch = "x86_64")]
pub type MatcherAVX512 = MatcherImpl<PrefilterAVX512, SmithWatermanAVX512>;
#[cfg(target_arch = "x86_64")]
pub type MatcherAVXU8 = MatcherImpl<PrefilterAVX, SmithWatermanAVXU8>;
#[cfg(target_arch = "x86_64")]
pub type MatcherAVX = MatcherImpl<PrefilterAVX, SmithWatermanAVX>;
#[cfg(target_arch = "x86_64")]
pub type MatcherSSEU8 = MatcherImpl<PrefilterSSE, SmithWatermanSSEU8>;
#[cfg(target_arch = "x86_64")]
pub type MatcherSSE = MatcherImpl<PrefilterSSE, SmithWatermanSSE>;
#[cfg(target_arch = "aarch64")]
pub type MatcherNEONU8 = MatcherImpl<PrefilterNEON, SmithWatermanNEONU8>;
#[cfg(target_arch = "aarch64")]
pub type MatcherNEON = MatcherImpl<PrefilterNEON, SmithWatermanNEON>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type MatcherWasmU8 = MatcherImpl<PrefilterWasm, SmithWatermanWasmU8>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type MatcherWasm = MatcherImpl<PrefilterWasm, SmithWatermanWasm>;
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
pub type MatcherScalar = MatcherImpl<PrefilterScalar, SmithWatermanScalar>;
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
pub type MatcherScalarU8 = MatcherImpl<PrefilterScalar, SmithWatermanScalarU8>;

#[derive(Debug, Clone)]
pub enum MatcherBackend {
    #[cfg(target_arch = "x86_64")]
    AVX512U8(MatcherAVX512U8),
    #[cfg(target_arch = "x86_64")]
    AVX512(MatcherAVX512),
    #[cfg(target_arch = "x86_64")]
    AVXU8(MatcherAVXU8),
    #[allow(clippy::upper_case_acronyms)]
    #[cfg(target_arch = "x86_64")]
    AVX(MatcherAVX),
    #[cfg(target_arch = "x86_64")]
    SSEU8(MatcherSSEU8),
    #[allow(clippy::upper_case_acronyms)]
    #[cfg(target_arch = "x86_64")]
    SSE(MatcherSSE),
    #[cfg(target_arch = "aarch64")]
    NEONU8(MatcherNEONU8),
    #[cfg(target_arch = "aarch64")]
    NEON(MatcherNEON),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    WasmU8(MatcherWasmU8),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm(MatcherWasm),
    // simd128 cannot be runtime-detected, so when it's statically enabled the scalar
    // fallback is unreachable
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    )))]
    ScalarU8(MatcherScalarU8),
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    )))]
    Scalar(MatcherScalar),

    // Literal matching backends (exact / prefix / suffix / substring). Selected when
    // `Config::matching` is not `Matching::Fuzzy`.
    #[cfg(target_arch = "x86_64")]
    LiteralAVX512(LiteralAVX512),
    #[cfg(target_arch = "x86_64")]
    LiteralAVX(LiteralAVX),
    #[cfg(target_arch = "x86_64")]
    LiteralSSE(LiteralSSE),
    #[cfg(target_arch = "aarch64")]
    LiteralNEON(LiteralNEON),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    LiteralWasm(LiteralWasm),
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    )))]
    LiteralScalar(LiteralScalar),
}

/// Implements [`Specialized`] for one concrete backend, attaching the
/// backend's `#[target_feature]` to each method (see the [`Specialized`] docs
/// for how this establishes the feature boundary). The bodies forward to the
/// `#[inline(always)]` loop helpers on [`MatcherImpl`]. Scalar backends omit
/// the feature argument.
macro_rules! impl_specialized {
    ($prefilter:ty, $smith_waterman:ty $(, target_feature = $feature:literal)?) => {
        impl Specialized for MatcherImpl<$prefilter, $smith_waterman> {
            #[inline]
            $(#[target_feature(enable = $feature)])?
            unsafe fn build(needle: &str, config: &Config) -> Self {
                Self::new(needle, config)
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_list<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystacks: &[H],
                haystack_index_offset: u32,
                matches: &mut Vec<Match>,
            ) {
                self.match_list_into_impl::<TYPOS, UNICODE, H>(
                    haystacks,
                    haystack_index_offset,
                    matches,
                )
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_list_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystacks: &[H],
            ) -> Vec<MatchIndices> {
                self.match_list_indices_impl::<TYPOS, UNICODE, H>(haystacks)
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_one<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystack: H,
                index: u32,
            ) -> Option<Match> {
                self.match_one_impl::<TYPOS, UNICODE, H>(haystack, index)
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_one_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystack: H,
                index: u32,
            ) -> Option<MatchIndices> {
                self.match_one_indices_impl::<TYPOS, UNICODE, H>(haystack, index)
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterAVX512,
    SmithWatermanAVX512U8,
    target_feature = "avx512f,avx512bw,avx512vbmi,bmi1,bmi2"
);
#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterAVX512,
    SmithWatermanAVX512,
    target_feature = "avx512f,avx512bw,bmi1,bmi2"
);
#[cfg(target_arch = "x86_64")]
impl_specialized!(PrefilterAVX, SmithWatermanAVXU8, target_feature = "avx2");
#[cfg(target_arch = "x86_64")]
impl_specialized!(PrefilterAVX, SmithWatermanAVX, target_feature = "avx2");
#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterSSE,
    SmithWatermanSSEU8,
    target_feature = "sse2,ssse3,sse4.1"
);
#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterSSE,
    SmithWatermanSSE,
    target_feature = "sse2,ssse3,sse4.1"
);
#[cfg(target_arch = "aarch64")]
impl_specialized!(PrefilterNEON, SmithWatermanNEONU8, target_feature = "neon");
#[cfg(target_arch = "aarch64")]
impl_specialized!(PrefilterNEON, SmithWatermanNEON, target_feature = "neon");
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl_specialized!(
    PrefilterWasm,
    SmithWatermanWasmU8,
    target_feature = "simd128"
);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl_specialized!(PrefilterWasm, SmithWatermanWasm, target_feature = "simd128");
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
impl_specialized!(PrefilterScalar, SmithWatermanScalarU8);
#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
impl_specialized!(PrefilterScalar, SmithWatermanScalar);
