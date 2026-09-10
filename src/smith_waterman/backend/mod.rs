//! SIMD backend abstraction for the Smith-Waterman kernel.
//!
//! Each backend exposes its native vector width via [`Backend::LANES`] and
//! three associated vector types: [`BytesVec`] for haystack bytes (LANES bytes
//! wide), [`MaskVec`] for boolean comparison results, and [`ScoreVec`] for the
//! score matrix (LANES × u16 or LANES × u8). Keeping masks as a distinct type
//! lets AVX-512 carry them as native `__mmask*` bitmasks rather than
//! 64-byte vectors, while other backends can alias [`MaskVec`] to their
//! [`BytesVec`] type with no loss.
//!
//! The kernel itself is generic over [`Backend`] and the lane count threads
//! through chunk arithmetic, the matrix stride, the alignment-path iterator,
//! and the horizontal-gap propagation unroll.
//!
//! Available backends:
//!   - AVX-512: LANES = 32/64 (scoring u16 x 32 = 512-bit or u8 x 64 = 512-bit)
//!   - AVX2:    LANES = 16/32 (scoring u16 x 16 = 256-bit or u8 x 32 = 256-bit)
//!   - SSE:     LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - NEON:    LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - WASM:    LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - Scalar:  LANES = 8/16 (fallback for non-SIMD systems)

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(
    test,
    not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    ))
))]
mod scalar;
#[cfg(target_arch = "x86_64")]
mod sse;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

#[cfg(target_arch = "x86_64")]
pub use avx::{BackendAVX, BackendAVXU8};
#[cfg(target_arch = "x86_64")]
pub use avx512::{BackendAVX512, BackendAVX512U8};
#[cfg(target_arch = "aarch64")]
pub use neon::{BackendNEON, BackendNEONU8};
#[cfg(any(
    test,
    not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    ))
))]
pub use scalar::{BackendScalar8, BackendScalar16U8};
#[cfg(target_arch = "x86_64")]
pub use sse::{BackendSSE, BackendSSEU8};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm::{BackendWasm, BackendWasmU8};

/// A SIMD backend for Smith-Waterman matching that supports variable-width
/// lanes (8, 16, 32, 64) and score primitives (u8, u16).
///
/// `LANES` is the number of cells per chunk in the score matrix. The matching
/// [`BytesVec`] holds `LANES` meaningful bytes (one per lane); comparisons
/// produce a [`MaskVec`] which can be widened element-wise to the [`ScoreVec`]
/// via [`Backend::widen_mask`].
pub trait Backend: Sized + core::fmt::Debug + Clone + 'static {
    const LANES: usize;
    /// Size of a single score lane in bytes. 1 for u8-element backends,
    /// 2 for u16-element backends. Used by the alignment-path iterator to
    /// read individual cells from the matrix's raw byte view.
    const LANE_BYTES: usize;
    type Bytes: BytesVec<Mask = Self::Mask>;
    type Mask: MaskVec;
    type Score: ScoreVec;

    /// Whether this backend's required CPU features are available at runtime
    fn is_available() -> bool;

    /// Widen a comparison [`MaskVec`] into a [`ScoreVec`] with each lane set
    /// to `0xFF` (u8 score) or `0xFFFF` (u16 score) where the mask is true and
    /// zero elsewhere.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score;

    /// Propagate horizontal (left-direction) gaps across a score row.
    ///
    /// The number of unrolled stages is fixed per backend: 8-lane backends do
    /// 3 stages (shifts of 1, 2, 4 lanes), 16-lane backends do 4 stages (1, 2,
    /// 4, 8), 32-lane backends do 5 stages, and 64-lane backends do 6 stages.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn propagate_horizontal_gaps(
        row: Self::Score,
        adjacent_row: Self::Score,
        match_mask: Self::Score,
        adjacent_match_mask: Self::Score,
        gap_open_penalty: Self::Score,
        gap_extend_penalty: Self::Score,
    ) -> Self::Score;

    /// Propagate Unicode-aware horizontal gaps across a score row.
    ///
    /// UTF-8 body bytes are transport lanes: they can carry scores and pending
    /// gap-open state without charging a gap step. Scalar-end lanes charge the
    /// affine gap cost. The return value includes the row scores and the
    /// pending gap-open state needed by the next haystack chunk.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    #[allow(clippy::too_many_arguments)]
    unsafe fn propagate_horizontal_unicode_gaps(
        row: Self::Score,
        adjacent_row: Self::Score,
        pending_gap_open_mask: Self::Score,
        adjacent_pending_gap_open_mask: Self::Score,
        continuation_gap_extend_penalty: Self::Score,
        adjacent_continuation_gap_extend_penalty: Self::Score,
        scalar_end_mask: Self::Score,
        adjacent_scalar_end_mask: Self::Score,
        gap_open_penalty: Self::Score,
        gap_extend_penalty: Self::Score,
    ) -> (Self::Score, Self::Score);
}

/// A byte-wide vector with `LANES` meaningful bytes
pub trait BytesVec: Copy + core::fmt::Debug {
    /// The backend's mask type, produced by comparison operations.
    type Mask: MaskVec;

    /// Broadcast `value` into every byte lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u8) -> Self;

    /// Per-lane equality, producing a true bit in each lane where equal.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn eq(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned greater-than, producing a true bit where
    /// `self > other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn gt(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned less-than, producing a true bit where `self < other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn lt(self, other: Self) -> Self::Mask;

    /// Load up to `LANES` bytes from `data + start`. `len` is the total length
    /// of the buffer pointed to by `data`.
    ///
    /// # Safety
    /// Caller must guarantee `data` points to at least `len` bytes and the
    /// backend's target features are enabled.
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self;

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8>;
}

/// A boolean mask over `LANES` lanes, produced by [`BytesVec`] comparisons.
///
/// Each lane holds a logical true/false. Non-AVX-512 backends typically alias
/// this to their [`BytesVec`] type and represent the mask as 0xFF/0x00 per
/// byte. AVX-512 uses a native `__mmask64` bitmask.
pub trait MaskVec: Copy + core::fmt::Debug {
    /// False in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Per-lane logical AND.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Per-lane logical OR.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn or(self, other: Self) -> Self;

    /// Per-lane logical NOT.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn not(self) -> Self;

    /// Whether every lane is false.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn is_zero(self) -> bool;

    /// Shift right by 1 lane, filling lane 0 with the highest meaningful lane
    /// of `prev`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self;

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool>;
}

/// A score-wide vector holding `LANES` u8 or u16 elements.
pub trait ScoreVec: Copy + core::fmt::Debug {
    /// Zero in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Broadcast `value` into every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u16) -> Self;

    /// `value` in lane 0, zero in all other lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn first_lane(value: u16) -> Self;

    /// Per-lane unsigned max
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn max(self, other: Self) -> Self;

    /// Horizontal unsigned max across all lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn horizontal_max(self) -> u16;

    /// Per-lane saturating add
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn add(self, other: Self) -> Self;

    /// Per-lane saturating subtract
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn subs(self, other: Self) -> Self;

    /// Bitwise AND
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Shift right by `L` lanes, filling the low `L` lanes with the high `L`
    /// lanes of `prev`. `L` must be in `0..=LANES`. `L == LANES` returns
    /// `prev` unmodified.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self;

    /// Index of the lowest lane equal to `search`, or `LANES` if absent
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn find_lane(self, search: u16) -> usize;

    // ----- Test-only helpers ---------------------------------------------

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16>;
}

#[cfg(test)]
mod tests;
