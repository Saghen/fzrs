//! Target-feature-specific instantiations of the literal matcher, mirroring
//! `src/matcher/backend.rs`. Each backend attaches its `#[target_feature]` to
//! the [`Specialized`] methods, which forward to the `#[inline(always)]`
//! helpers on [`LiteralImpl`].

use super::algo::LiteralImpl;
use crate::matcher::algo::Specialized;
use crate::{Config, Match, MatchIndices};
use alloc::vec::Vec;

#[cfg(target_arch = "aarch64")]
use crate::prefilter::backend::PrefilterNEONBackend;
#[cfg(any(
    test,
    not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    ))
))]
use crate::prefilter::backend::PrefilterScalarBackend;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::prefilter::backend::PrefilterWasmBackend;
#[cfg(target_arch = "x86_64")]
use crate::prefilter::backend::{PrefilterAVX512Backend, PrefilterAVXBackend, PrefilterSSEBackend};

#[cfg(target_arch = "x86_64")]
pub(crate) type LiteralAVX512 = LiteralImpl<PrefilterAVX512Backend>;
#[cfg(target_arch = "x86_64")]
pub(crate) type LiteralAVX = LiteralImpl<PrefilterAVXBackend>;
#[cfg(target_arch = "x86_64")]
pub(crate) type LiteralSSE = LiteralImpl<PrefilterSSEBackend>;
#[cfg(target_arch = "aarch64")]
pub(crate) type LiteralNEON = LiteralImpl<PrefilterNEONBackend>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) type LiteralWasm = LiteralImpl<PrefilterWasmBackend>;
#[cfg(any(
    test,
    not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    ))
))]
pub(crate) type LiteralScalar = LiteralImpl<PrefilterScalarBackend>;

/// Implements [`Specialized`] for one literal backend. `TYPOS` is ignored
/// (literal matching has no typo tolerance); `UNICODE` selects the byte-level
/// ASCII path or the per-codepoint unicode path.
macro_rules! impl_specialized_literal {
    ($backend:ty $(, target_feature = $feature:literal)?) => {
        impl Specialized for LiteralImpl<$backend> {
            #[inline]
            $(#[target_feature(enable = $feature)])?
            unsafe fn build(needle: &str, config: &Config) -> Self {
                unsafe { Self::new(needle, config) }
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_list<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystacks: &[H],
                haystack_index_offset: u32,
                matches: &mut Vec<Match>,
            ) {
                unsafe {
                    self.match_list_impl::<UNICODE, H>(haystacks, haystack_index_offset, matches)
                }
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_list_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystacks: &[H],
            ) -> Vec<MatchIndices> {
                unsafe { self.match_list_indices_impl::<UNICODE, H>(haystacks) }
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_one<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystack: H,
                index: u32,
            ) -> Option<Match> {
                unsafe { self.match_one_impl::<UNICODE, H>(haystack, index) }
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_one_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystack: H,
                index: u32,
            ) -> Option<MatchIndices> {
                unsafe { self.match_one_indices_impl::<UNICODE, H>(haystack, index) }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
impl_specialized_literal!(
    PrefilterAVX512Backend,
    target_feature = "avx512f,avx512bw,bmi1,bmi2"
);
#[cfg(target_arch = "x86_64")]
impl_specialized_literal!(PrefilterAVXBackend, target_feature = "avx2");
#[cfg(target_arch = "x86_64")]
impl_specialized_literal!(PrefilterSSEBackend, target_feature = "sse2");
#[cfg(target_arch = "aarch64")]
impl_specialized_literal!(PrefilterNEONBackend, target_feature = "neon");
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl_specialized_literal!(PrefilterWasmBackend, target_feature = "simd128");
#[cfg(any(
    test,
    not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    ))
))]
impl_specialized_literal!(PrefilterScalarBackend);

#[cfg(test)]
mod backend_parity {
    use crate::matcher::algo::Specialized;
    use crate::{Config, Matching, SortStrategy};

    /// Runs one needle/haystack through a specialized literal backend,
    /// returning the observable result of both `match_one` and
    /// `match_one_indices`. `UNICODE` is chosen the same way the real
    /// dispatch chooses it (`respects_unicode_for`), so non-ASCII needles
    /// exercise the codepoint path on every backend.
    #[allow(clippy::type_complexity)]
    unsafe fn probe<T: Specialized>(
        needle: &str,
        haystack: &str,
        config: &Config,
    ) -> (Option<(u16, bool)>, Option<Vec<u32>>) {
        let mut matcher = unsafe { T::build(needle, config) };
        let (m, i) = if config.unicode.respects_unicode_for(needle) {
            (
                unsafe { matcher.match_one::<0, true, &str>(haystack, 0) },
                unsafe { matcher.match_one_indices::<0, true, &str>(haystack, 0) },
            )
        } else {
            (
                unsafe { matcher.match_one::<0, false, &str>(haystack, 0) },
                unsafe { matcher.match_one_indices::<0, false, &str>(haystack, 0) },
            )
        };
        (m.map(|m| (m.score, m.exact)), i.map(|i| i.indices))
    }

    #[test]
    fn simd_backends_agree_with_scalar() {
        use crate::literal::LiteralScalar;

        let cases: &[(&str, &str)] = &[
            ("bar", "foobar"),
            ("bar", "foo_bar"),
            ("bar", "barbarbar"),
            ("ab", "ab_ab"),
            ("foo", "foo"),
            ("x", &"y".repeat(200)),
            ("needle", "xxneedlexxneedle_needle"),
            ("é다😀", "xxé다😀yyé다😀"),
            ("다", "가나다라마"),
            ("z", "abcdefghijklmnopqrstuvwxyz"),
            ("FoO", "prefix_FoO_FOO_foo"),
            // Two-byte prefilter stress: common first byte, len 1/2, first == last, match at the
            // end, and overlapping occurrences.
            ("a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            ("ab", "abababababababababababab"),
            ("aa", "baaab"),
            ("aaaa", "xaaaaaaaax"),
            ("aaa", "aaaaa"),
            ("bar", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxbar"),
            ("ba", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaba"),
            ("foobar", "foobatefoobarfoobar"),
            // Unicode codepoint path with case folding: mixed-case occurrences, script mixes, and
            // the Cherokee hybrid (E1 8E A0 / EA AD B0) that per-byte matching would wrongly
            // accept.
            ("é", "xÉyéZÉ"),
            ("café", "un CAFÉ, deux cafés"),
            ("Ꭰ", "\u{1b70}Ꭰꭰ\u{1b70}"),
            ("иха", "МУХА_ИХА_иха"),
            ("αβ", "ΑΒβα_αβ"),
        ];

        for &(needle, haystack) in cases {
            for matching in [
                Matching::Exact,
                Matching::Prefix,
                Matching::Suffix,
                Matching::Substring,
            ] {
                let config = Config::default()
                    .matching(matching)
                    .sort(SortStrategy::IndexAsc);
                let expected = unsafe { probe::<LiteralScalar>(needle, haystack, &config) };

                #[cfg(target_arch = "x86_64")]
                {
                    use crate::literal::{LiteralAVX, LiteralAVX512, LiteralSSE};
                    if LiteralAVX512::is_available() {
                        assert_eq!(
                            unsafe { probe::<LiteralAVX512>(needle, haystack, &config) },
                            expected,
                            "AVX-512 mismatch: needle={needle:?} haystack={haystack:?} {matching:?}"
                        );
                    }
                    if LiteralAVX::is_available() {
                        assert_eq!(
                            unsafe { probe::<LiteralAVX>(needle, haystack, &config) },
                            expected,
                            "AVX2 mismatch: needle={needle:?} haystack={haystack:?} {matching:?}"
                        );
                    }
                    if LiteralSSE::is_available() {
                        assert_eq!(
                            unsafe { probe::<LiteralSSE>(needle, haystack, &config) },
                            expected,
                            "SSE mismatch: needle={needle:?} haystack={haystack:?} {matching:?}"
                        );
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    use crate::literal::LiteralNEON;
                    if LiteralNEON::is_available() {
                        assert_eq!(
                            unsafe { probe::<LiteralNEON>(needle, haystack, &config) },
                            expected,
                            "NEON mismatch: needle={needle:?} haystack={haystack:?} {matching:?}"
                        );
                    }
                }

                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    use crate::literal::LiteralWasm;
                    if LiteralWasm::is_available() {
                        assert_eq!(
                            unsafe { probe::<LiteralWasm>(needle, haystack, &config) },
                            expected,
                            "WASM mismatch: needle={needle:?} haystack={haystack:?} {matching:?}"
                        );
                    }
                }
            }
        }
    }
}
