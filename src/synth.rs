//! Dual software synthesis engines for the Rareware GBA Sound Engine.
//!
//! Provides two synthesis pipelines:
//! - [`AccuracySynth`]: 100% bit-exact emulation of all four Rareware GBA ARM7TDMI IWRAM
//!   mixer revisions (`StandardSingleClamp`, `StandardDualHalfClamp`, `StandardDualByteClamp`,
//!   and `PackedFastPath`), verified byte-for-byte (`2,688 / 2,688` bytes) against real
//!   ARM7TDMI execution in headless `libmgba`.
//! - [`StudioSynth`]: High-fidelity 48,000 Hz `f32` stereo renderer with a compile-time
//!   normalized 1,024-phase 6-tap Lanczos-3 windowed-sinc polyphase LUT and equal-power
//!   stereo panning.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::freq::{RARE_DEFAULT_MIX_RATE, RARE_FRAC_BITS, RARE_FRAC_MASK, RARE_FRAC_ONE};
use crate::sample::RareSampleView;
use crate::version::RareAudioVersion;
use crate::voice::RareVoice;

/// Number of fractional polyphase steps in [`StudioSynth`]'s compile-time Lanczos-3 LUT.
const LANCZOS_PHASES: usize = 1024;

/// Compile-time normalized 1,024-phase 6-tap Lanczos-3 polyphase lookup table
/// (`1025 * 6 * 4 = 24,600` bytes — L1D-cache resident).
static LANCZOS3_LUT: [[f32; 6]; LANCZOS_PHASES + 1] = build_lanczos3_lut();

/// Build the normalized 6-tap Lanczos-3 polyphase lookup table at compile time.
const fn build_lanczos3_lut() -> [[f32; 6]; LANCZOS_PHASES + 1] {
    let mut table = [[0.0f32; 6]; LANCZOS_PHASES + 1];
    let mut p = 0usize;
    while p <= LANCZOS_PHASES {
        let frac = (p as f64) / (LANCZOS_PHASES as f64);
        let mut weights = [0.0f64; 6];
        let mut sum = 0.0f64;
        let mut idx = 0usize;
        while idx < 6 {
            let k = (idx as i32) - 2; // taps -2..=+3
            let w = lanczos_kernel(frac - (k as f64), 3.0);
            weights[idx] = w;
            sum += w;
            idx += 1;
        }
        let inv_sum = if sum > 1e-12 { 1.0 / sum } else { 0.0 };
        let mut j = 0usize;
        while j < 6 {
            table[p][j] = (weights[j] * inv_sum) as f32;
            j += 1;
        }
        p += 1;
    }
    table
}

/// Compile-time `sin(x)` using range reduction and a 9-term Taylor polynomial.
const fn const_sin(x: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    const TWO_PI: f64 = 2.0 * PI;
    let mut y = x;
    while y > PI {
        y -= TWO_PI;
    }
    while y < -PI {
        y += TWO_PI;
    }
    let y2 = y * y;
    let mut term = y;
    let mut sum = y;
    let mut n = 1i32;
    while n <= 9 {
        term = -term * y2 / ((2 * n) as f64 * (2 * n + 1) as f64);
        sum += term;
        n += 1;
    }
    sum
}

/// Compile-time Lanczos windowed-sinc kernel `sinc(x) * sinc(x / a)` for `|x| < a`.
const fn lanczos_kernel(x: f64, a: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1e-12 {
        return 1.0;
    }
    if ax >= a {
        return 0.0;
    }
    let pi_x = core::f64::consts::PI * x;
    let pi_x_a = pi_x / a;
    (const_sin(pi_x) / pi_x) * (const_sin(pi_x_a) / pi_x_a)
}

/// Compute square root of an `f64` value (`#![no_std]` compatible).
#[inline]
#[must_use]
pub fn sqrt_f64(val: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        val.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        if val < 0.0 {
            return f64::NAN;
        }
        if val == 0.0 || val.is_infinite() || val.is_nan() {
            return val;
        }
        let bits = val.to_bits();
        let mut x = f64::from_bits((bits >> 1) + (1023u64 << 51));
        if x <= 0.0 || x.is_nan() {
            x = val;
        }
        for _ in 0..6 {
            x = 0.5 * (x + val / x);
        }
        x
    }
}

/// Bit-exact software mixer emulating all four Rareware GBA ARM7TDMI IWRAM mixer revisions.
///
/// Mirrors the real hardware DirectSound A mono pipeline:
/// 1. 23-bit fractional linear interpolation (`s0 + ((frac * (s1 - s0)) >> 23)` with signed `asr #23`).
/// 2. Signed volume scaling (`(interp * vol) >> 7`).
/// 3. Revision-accurate 16-bit scratch accumulation (including *Banjo-Pilot*'s lookahead-guarded
///    2x-packed `add r2, sl, r2, lsl #16` first-voice negative-low-halfword `-1` carry bleed).
/// 4. Exact loop wrap (`pos -= loop_length`) and one-shot early return (`active = false`).
/// 5. DirectSound A `i16 -> i8` saturation clamping to `[-128, +127]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracySynth {
    /// Hardware DirectSound A mixing rate in Hz (default `13,379` Hz).
    pub sample_rate: u32,
    /// Rareware ARM mixer revision to emulate.
    pub version: RareAudioVersion,
    /// Active voices managed by this synthesizer instance (`alloc` feature).
    #[cfg(feature = "alloc")]
    pub voices: Vec<RareVoice>,
}

impl Default for AccuracySynth {
    fn default() -> Self {
        Self::new(RARE_DEFAULT_MIX_RATE)
    }
}

impl AccuracySynth {
    /// Create a new `AccuracySynth` configured for Rareware's default 13,379 Hz rate
    /// and [`RareAudioVersion::StandardSingleClamp`].
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self::with_version(sample_rate, RareAudioVersion::StandardSingleClamp)
    }

    /// Create a new `AccuracySynth` configured for a specific [`RareAudioVersion`].
    #[must_use]
    pub fn with_version(sample_rate: u32, version: RareAudioVersion) -> Self {
        Self {
            sample_rate,
            version,
            #[cfg(feature = "alloc")]
            voices: Vec::new(),
        }
    }

    /// Standard per-sample `strh`/`ldrsh` mixer kernel specialized on `IS_FIRST`,
    /// using interior-run chunking and a 1:1 pitch SIMD fast path while remaining
    /// 100% bit-exact with Rareware's ARM7TDMI inner loop.
    #[inline(always)]
    fn mix_standard_impl<const IS_FIRST: bool>(
        sample: &RareSampleView<'_>,
        voice: &mut RareVoice,
        acc: &mut [i16],
    ) {
        const FRAC_MASK_U64: u64 = RARE_FRAC_MASK as u64;

        let n = acc.len();
        let pcm = sample.pcm;
        let pcm_len = pcm.len();
        let sample_len = sample.sample_len;
        let loop_len = sample.loop_length as usize;
        let is_looped = voice.playback_mode.is_looped() && loop_len > 0;
        let step = voice.step_q23;
        let step_u64 = step as u64;
        let vol = voice.volume;

        let safe_end = sample_len.min(pcm_len);
        let limit_pos = sample_len.min(pcm_len.saturating_sub(1));

        let mut pos = voice.pos;
        let mut frac = voice.frac;
        let mut offset = 0usize;

        while offset < n {
            // 1:1 pitch fast path: contiguous slice enables full SIMD auto-vectorization
            if step == RARE_FRAC_ONE && frac == 0 && pos < safe_end {
                let run = (safe_end - pos).min(n - offset);
                let src = &pcm[pos..pos + run];
                let dst = &mut acc[offset..offset + run];

                if IS_FIRST {
                    if vol == 128 {
                        for (&b, slot) in src.iter().zip(dst.iter_mut()) {
                            *slot = b as i8 as i16;
                        }
                    } else {
                        for (&b, slot) in src.iter().zip(dst.iter_mut()) {
                            *slot = ((b as i8 as i32).wrapping_mul(vol) >> 7) as i16;
                        }
                    }
                } else if vol == 128 {
                    for (&b, slot) in src.iter().zip(dst.iter_mut()) {
                        *slot = slot.wrapping_add(b as i8 as i16);
                    }
                } else {
                    for (&b, slot) in src.iter().zip(dst.iter_mut()) {
                        *slot = slot.wrapping_add(((b as i8 as i32).wrapping_mul(vol) >> 7) as i16);
                    }
                }

                pos += run;
                offset += run;
                if pos >= sample_len {
                    if is_looped {
                        pos = pos.saturating_sub(loop_len);
                    } else {
                        voice.active = false;
                        break;
                    }
                }
                continue;
            }

            // Resampled Q9.23 interior fast path: single 64-bit phase accumulator,
            // zero per-sample loop-wrap/end checks, and direct adjacent-pair reads.
            if step > 0 && step <= 0x7FFF_FFFF && frac <= RARE_FRAC_MASK && pos < limit_pos {
                let ph0 = ((pos as u64) << RARE_FRAC_BITS) | (frac as u64);
                let limit_ph = (limit_pos as u64) << RARE_FRAC_BITS;
                let k_max = (limit_ph - 1 - ph0) / step_u64;
                let run = ((k_max + 1) as usize).min(n - offset);

                let src = &pcm[..limit_pos + 1];
                let dst = &mut acc[offset..offset + run];
                let mut ph = ph0;
                let step2 = step_u64 * 2;
                let step3 = step_u64 * 3;
                let step4 = step_u64 * 4;

                let (chunks, rem) = dst.as_chunks_mut::<4>();
                if IS_FIRST && vol == 128 {
                    for chunk in chunks {
                        let p0 = ph;
                        let p1 = ph + step_u64;
                        let p2 = ph + step2;
                        let p3 = ph + step3;
                        ph += step4;

                        let i0 = (p0 >> RARE_FRAC_BITS) as usize;
                        let a0 = &src[i0..i0 + 2];
                        let s0 = a0[0] as i8 as i32;
                        let d0 = (a0[1] as i8 as i32) - s0;
                        chunk[0] = (s0 + (((p0 & FRAC_MASK_U64) as i32).wrapping_mul(d0) >> 23)) as i16;

                        let i1 = (p1 >> RARE_FRAC_BITS) as usize;
                        let a1 = &src[i1..i1 + 2];
                        let s1 = a1[0] as i8 as i32;
                        let d1 = (a1[1] as i8 as i32) - s1;
                        chunk[1] = (s1 + (((p1 & FRAC_MASK_U64) as i32).wrapping_mul(d1) >> 23)) as i16;

                        let i2 = (p2 >> RARE_FRAC_BITS) as usize;
                        let a2 = &src[i2..i2 + 2];
                        let s2 = a2[0] as i8 as i32;
                        let d2 = (a2[1] as i8 as i32) - s2;
                        chunk[2] = (s2 + (((p2 & FRAC_MASK_U64) as i32).wrapping_mul(d2) >> 23)) as i16;

                        let i3 = (p3 >> RARE_FRAC_BITS) as usize;
                        let a3 = &src[i3..i3 + 2];
                        let s3 = a3[0] as i8 as i32;
                        let d3 = (a3[1] as i8 as i32) - s3;
                        chunk[3] = (s3 + (((p3 & FRAC_MASK_U64) as i32).wrapping_mul(d3) >> 23)) as i16;
                    }
                    for slot in rem {
                        let idx = (ph >> RARE_FRAC_BITS) as usize;
                        let pair = &src[idx..idx + 2];
                        let s0 = pair[0] as i8 as i32;
                        let s1 = pair[1] as i8 as i32;
                        let f = (ph & FRAC_MASK_U64) as i32;
                        *slot = (s0 + (f.wrapping_mul(s1 - s0) >> 23)) as i16;
                        ph += step_u64;
                    }
                } else if IS_FIRST {
                    for chunk in chunks {
                        let p0 = ph;
                        let p1 = ph + step_u64;
                        let p2 = ph + step2;
                        let p3 = ph + step3;
                        ph += step4;

                        let i0 = (p0 >> RARE_FRAC_BITS) as usize;
                        let a0 = &src[i0..i0 + 2];
                        let s0 = a0[0] as i8 as i32;
                        let d0 = (a0[1] as i8 as i32) - s0;
                        let v0 = s0 + (((p0 & FRAC_MASK_U64) as i32).wrapping_mul(d0) >> 23);
                        chunk[0] = (v0.wrapping_mul(vol) >> 7) as i16;

                        let i1 = (p1 >> RARE_FRAC_BITS) as usize;
                        let a1 = &src[i1..i1 + 2];
                        let s1 = a1[0] as i8 as i32;
                        let d1 = (a1[1] as i8 as i32) - s1;
                        let v1 = s1 + (((p1 & FRAC_MASK_U64) as i32).wrapping_mul(d1) >> 23);
                        chunk[1] = (v1.wrapping_mul(vol) >> 7) as i16;

                        let i2 = (p2 >> RARE_FRAC_BITS) as usize;
                        let a2 = &src[i2..i2 + 2];
                        let s2 = a2[0] as i8 as i32;
                        let d2 = (a2[1] as i8 as i32) - s2;
                        let v2 = s2 + (((p2 & FRAC_MASK_U64) as i32).wrapping_mul(d2) >> 23);
                        chunk[2] = (v2.wrapping_mul(vol) >> 7) as i16;

                        let i3 = (p3 >> RARE_FRAC_BITS) as usize;
                        let a3 = &src[i3..i3 + 2];
                        let s3 = a3[0] as i8 as i32;
                        let d3 = (a3[1] as i8 as i32) - s3;
                        let v3 = s3 + (((p3 & FRAC_MASK_U64) as i32).wrapping_mul(d3) >> 23);
                        chunk[3] = (v3.wrapping_mul(vol) >> 7) as i16;
                    }
                    for slot in rem {
                        let idx = (ph >> RARE_FRAC_BITS) as usize;
                        let pair = &src[idx..idx + 2];
                        let s0 = pair[0] as i8 as i32;
                        let s1 = pair[1] as i8 as i32;
                        let f = (ph & FRAC_MASK_U64) as i32;
                        let interp = s0 + (f.wrapping_mul(s1 - s0) >> 23);
                        *slot = (interp.wrapping_mul(vol) >> 7) as i16;
                        ph += step_u64;
                    }
                } else {
                    for chunk in chunks {
                        let p0 = ph;
                        let p1 = ph + step_u64;
                        let p2 = ph + step2;
                        let p3 = ph + step3;
                        ph += step4;

                        let i0 = (p0 >> RARE_FRAC_BITS) as usize;
                        let a0 = &src[i0..i0 + 2];
                        let s0 = a0[0] as i8 as i32;
                        let d0 = (a0[1] as i8 as i32) - s0;
                        let v0 = s0 + (((p0 & FRAC_MASK_U64) as i32).wrapping_mul(d0) >> 23);
                        chunk[0] = chunk[0].wrapping_add((v0.wrapping_mul(vol) >> 7) as i16);

                        let i1 = (p1 >> RARE_FRAC_BITS) as usize;
                        let a1 = &src[i1..i1 + 2];
                        let s1 = a1[0] as i8 as i32;
                        let d1 = (a1[1] as i8 as i32) - s1;
                        let v1 = s1 + (((p1 & FRAC_MASK_U64) as i32).wrapping_mul(d1) >> 23);
                        chunk[1] = chunk[1].wrapping_add((v1.wrapping_mul(vol) >> 7) as i16);

                        let i2 = (p2 >> RARE_FRAC_BITS) as usize;
                        let a2 = &src[i2..i2 + 2];
                        let s2 = a2[0] as i8 as i32;
                        let d2 = (a2[1] as i8 as i32) - s2;
                        let v2 = s2 + (((p2 & FRAC_MASK_U64) as i32).wrapping_mul(d2) >> 23);
                        chunk[2] = chunk[2].wrapping_add((v2.wrapping_mul(vol) >> 7) as i16);

                        let i3 = (p3 >> RARE_FRAC_BITS) as usize;
                        let a3 = &src[i3..i3 + 2];
                        let s3 = a3[0] as i8 as i32;
                        let d3 = (a3[1] as i8 as i32) - s3;
                        let v3 = s3 + (((p3 & FRAC_MASK_U64) as i32).wrapping_mul(d3) >> 23);
                        chunk[3] = chunk[3].wrapping_add((v3.wrapping_mul(vol) >> 7) as i16);
                    }
                    for slot in rem {
                        let idx = (ph >> RARE_FRAC_BITS) as usize;
                        let pair = &src[idx..idx + 2];
                        let s0 = pair[0] as i8 as i32;
                        let s1 = pair[1] as i8 as i32;
                        let f = (ph & FRAC_MASK_U64) as i32;
                        let interp = s0 + (f.wrapping_mul(s1 - s0) >> 23);
                        let scaled = (interp.wrapping_mul(vol) >> 7) as i16;
                        *slot = slot.wrapping_add(scaled);
                        ph += step_u64;
                    }
                }

                pos = (ph >> RARE_FRAC_BITS) as usize;
                frac = (ph & FRAC_MASK_U64) as u32;
                offset += run;
                if pos >= sample_len {
                    if is_looped {
                        pos = pos.saturating_sub(loop_len);
                    } else {
                        voice.active = false;
                        break;
                    }
                }
                continue;
            }

            // Boundary / missing-guard / zero-step single-sample fallback
            let (s0, s1) = sample.sample_pair_i32(pos);
            let interp = s0 + (((frac as i32).wrapping_mul(s1 - s0)) >> 23);
            let scaled = (interp.wrapping_mul(vol) >> 7) as i16;

            if IS_FIRST {
                acc[offset] = scaled;
            } else {
                acc[offset] = acc[offset].wrapping_add(scaled);
            }
            offset += 1;

            frac = frac.wrapping_add(step);
            pos += (frac >> RARE_FRAC_BITS) as usize;
            frac &= RARE_FRAC_MASK;

            if pos >= sample_len {
                if is_looped {
                    pos = pos.saturating_sub(loop_len);
                } else {
                    voice.active = false;
                    break;
                }
            }
        }

        voice.pos = pos;
        voice.frac = frac;
    }

    /// Mix a single voice into a 16-bit signed scratch accumulator (`acc`), matching the
    /// exact ARM7TDMI instructions of `version`.
    ///
    /// - When `is_first` is `true`, the voice overwrites `acc` (matching Rareware's first-voice
    ///   branch). On [`RareAudioVersion::PackedFastPath`], if `acc.len()` is even and the ARM
    ///   `umull`/`adds`/`adc` lookahead determines the voice will not reach `sample.sample_len`
    ///   within `acc.len()` samples, this executes *Banjo-Pilot*'s exact 2x-packed `str` path
    ///   (including the negative-low-halfword `-1` bleed on odd samples).
    /// - When `is_first` is `false`, the voice accumulates into `acc` via wrapping 16-bit addition.
    pub fn mix_voice_view_version(
        sample: &RareSampleView<'_>,
        voice: &mut RareVoice,
        acc: &mut [i16],
        is_first: bool,
        version: RareAudioVersion,
    ) {
        if !voice.active || sample.is_empty() || acc.is_empty() {
            return;
        }

        const FRAC_MASK_U64: u64 = RARE_FRAC_MASK as u64;

        let n = acc.len();
        let sample_len = sample.sample_len;
        let step = voice.step_q23;
        let vol = voice.volume;

        // Exact Banjo-Pilot ARM lookahead (`0x1c00` / `0x1cc4`):
        //   r3 = n << 9; (r1, r2) = umull(r3, step);
        //   adds r1, r1, frac << 9; adc r2, r2, pos; cmp r2, sample_end
        // Equivalent to: pos + ((n * step + frac) >> 23) < sample_len
        if is_first && version.has_packed_first_voice_fast_path() && (n & 1) == 0 {
            let total_q23 = (n as u64)
                .wrapping_mul(step as u64)
                .wrapping_add(voice.frac as u64);
            let end_pos = voice.pos.saturating_add((total_q23 >> RARE_FRAC_BITS) as usize);
            if end_pos < sample_len {
                let pcm = sample.pcm;

                if step == RARE_FRAC_ONE && voice.frac == 0 && voice.pos + n <= pcm.len() {
                    // 1:1 pitch 2x-packed fast path preserving exact ARM `-1` carry bleed
                    let (src_pairs, _) = pcm[voice.pos..voice.pos + n].as_chunks::<2>();
                    let (dst_pairs, _) = acc.as_chunks_mut::<2>();
                    for (s_pair, dst) in src_pairs.iter().zip(dst_pairs.iter_mut()) {
                        let scaled0 = (s_pair[0] as i8 as i32).wrapping_mul(vol) >> 7;
                        let scaled1 = (s_pair[1] as i8 as i32).wrapping_mul(vol) >> 7;
                        let packed = (scaled0 as u32).wrapping_add((scaled1 as u32) << 16);
                        dst[0] = (packed & 0xFFFF) as i16;
                        dst[1] = (packed >> 16) as i16;
                    }
                    voice.pos += n;
                    return;
                }

                if voice.frac <= RARE_FRAC_MASK
                    && step <= 0x7FFF_FFFF
                    && end_pos + 1 < pcm.len()
                {
                    // Guarded interior 2x-packed fast path: single 64-bit phase accumulator,
                    // 4-sample (2-pair) unrolling, zero per-sample bounds checks on `acc`,
                    // exact ARM `-1` carry bleed.
                    let src = &pcm[..end_pos + 2];
                    let step_u64 = step as u64;
                    let step2 = step_u64 * 2;
                    let step3 = step_u64 * 3;
                    let step4 = step_u64 * 4;
                    let mut ph = ((voice.pos as u64) << RARE_FRAC_BITS) | (voice.frac as u64);

                    let (chunks, rem) = acc.as_chunks_mut::<4>();
                    if vol == 128 {
                        for dst in chunks {
                            let p0 = ph;
                            let p1 = ph + step_u64;
                            let p2 = ph + step2;
                            let p3 = ph + step3;
                            ph += step4;

                            let i0 = (p0 >> RARE_FRAC_BITS) as usize;
                            let a0 = &src[i0..i0 + 2];
                            let s0 = a0[0] as i8 as i32;
                            let sc0 = s0 + (((p0 & FRAC_MASK_U64) as i32).wrapping_mul((a0[1] as i8 as i32) - s0) >> 23);

                            let i1 = (p1 >> RARE_FRAC_BITS) as usize;
                            let a1 = &src[i1..i1 + 2];
                            let s1 = a1[0] as i8 as i32;
                            let sc1 = s1 + (((p1 & FRAC_MASK_U64) as i32).wrapping_mul((a1[1] as i8 as i32) - s1) >> 23);

                            let packed0 = (sc0 as u32).wrapping_add((sc1 as u32) << 16);
                            dst[0] = (packed0 & 0xFFFF) as i16;
                            dst[1] = (packed0 >> 16) as i16;

                            let i2 = (p2 >> RARE_FRAC_BITS) as usize;
                            let a2 = &src[i2..i2 + 2];
                            let s2 = a2[0] as i8 as i32;
                            let sc2 = s2 + (((p2 & FRAC_MASK_U64) as i32).wrapping_mul((a2[1] as i8 as i32) - s2) >> 23);

                            let i3 = (p3 >> RARE_FRAC_BITS) as usize;
                            let a3 = &src[i3..i3 + 2];
                            let s3 = a3[0] as i8 as i32;
                            let sc3 = s3 + (((p3 & FRAC_MASK_U64) as i32).wrapping_mul((a3[1] as i8 as i32) - s3) >> 23);

                            let packed1 = (sc2 as u32).wrapping_add((sc3 as u32) << 16);
                            dst[2] = (packed1 & 0xFFFF) as i16;
                            dst[3] = (packed1 >> 16) as i16;
                        }
                    } else {
                        for dst in chunks {
                            let p0 = ph;
                            let p1 = ph + step_u64;
                            let p2 = ph + step2;
                            let p3 = ph + step3;
                            ph += step4;

                            let i0 = (p0 >> RARE_FRAC_BITS) as usize;
                            let a0 = &src[i0..i0 + 2];
                            let s0 = a0[0] as i8 as i32;
                            let sc0 = (s0 + (((p0 & FRAC_MASK_U64) as i32).wrapping_mul((a0[1] as i8 as i32) - s0) >> 23)).wrapping_mul(vol) >> 7;

                            let i1 = (p1 >> RARE_FRAC_BITS) as usize;
                            let a1 = &src[i1..i1 + 2];
                            let s1 = a1[0] as i8 as i32;
                            let sc1 = (s1 + (((p1 & FRAC_MASK_U64) as i32).wrapping_mul((a1[1] as i8 as i32) - s1) >> 23)).wrapping_mul(vol) >> 7;

                            let packed0 = (sc0 as u32).wrapping_add((sc1 as u32) << 16);
                            dst[0] = (packed0 & 0xFFFF) as i16;
                            dst[1] = (packed0 >> 16) as i16;

                            let i2 = (p2 >> RARE_FRAC_BITS) as usize;
                            let a2 = &src[i2..i2 + 2];
                            let s2 = a2[0] as i8 as i32;
                            let sc2 = (s2 + (((p2 & FRAC_MASK_U64) as i32).wrapping_mul((a2[1] as i8 as i32) - s2) >> 23)).wrapping_mul(vol) >> 7;

                            let i3 = (p3 >> RARE_FRAC_BITS) as usize;
                            let a3 = &src[i3..i3 + 2];
                            let s3 = a3[0] as i8 as i32;
                            let sc3 = (s3 + (((p3 & FRAC_MASK_U64) as i32).wrapping_mul((a3[1] as i8 as i32) - s3) >> 23)).wrapping_mul(vol) >> 7;

                            let packed1 = (sc2 as u32).wrapping_add((sc3 as u32) << 16);
                            dst[2] = (packed1 & 0xFFFF) as i16;
                            dst[3] = (packed1 >> 16) as i16;
                        }
                    }

                    for dst in rem.as_chunks_mut::<2>().0 {
                        let idx0 = (ph >> RARE_FRAC_BITS) as usize;
                        let pair0 = &src[idx0..idx0 + 2];
                        let s0_a = pair0[0] as i8 as i32;
                        let s1_a = pair0[1] as i8 as i32;
                        let f0 = (ph & FRAC_MASK_U64) as i32;
                        let interp0 = s0_a + (f0.wrapping_mul(s1_a - s0_a) >> 23);
                        let scaled0 = interp0.wrapping_mul(vol) >> 7;
                        ph += step_u64;

                        let idx1 = (ph >> RARE_FRAC_BITS) as usize;
                        let pair1 = &src[idx1..idx1 + 2];
                        let s0_b = pair1[0] as i8 as i32;
                        let s1_b = pair1[1] as i8 as i32;
                        let f1 = (ph & FRAC_MASK_U64) as i32;
                        let interp1 = s0_b + (f1.wrapping_mul(s1_b - s0_b) >> 23);
                        let scaled1 = interp1.wrapping_mul(vol) >> 7;
                        ph += step_u64;

                        // Authentic Banjo-Pilot ARM `add r2, sl, r2, lsl #16`:
                        // `scaled0 as u32` has 0xFFFF in upper 16 bits when `scaled0 < 0`,
                        // which subtracts 1 from `scaled1` in the upper halfword!
                        let packed = (scaled0 as u32).wrapping_add((scaled1 as u32) << 16);
                        dst[0] = (packed & 0xFFFF) as i16;
                        dst[1] = (packed >> 16) as i16;
                    }

                    voice.pos = (ph >> RARE_FRAC_BITS) as usize;
                    voice.frac = (ph & FRAC_MASK_U64) as u32;
                    return;
                }

                // Safe fallback if sample lacks trailing guard byte or has unmasked frac
                let mut pos = voice.pos;
                let mut frac = voice.frac;
                for dst in acc.as_chunks_mut::<2>().0 {
                    let (s0_a, s1_a) = sample.sample_pair_i32(pos);
                    let interp0 = s0_a + (((frac as i32).wrapping_mul(s1_a - s0_a)) >> 23);
                    let scaled0 = interp0.wrapping_mul(vol) >> 7;
                    frac = frac.wrapping_add(step);
                    pos += (frac >> RARE_FRAC_BITS) as usize;
                    frac &= RARE_FRAC_MASK;

                    let (s0_b, s1_b) = sample.sample_pair_i32(pos);
                    let interp1 = s0_b + (((frac as i32).wrapping_mul(s1_b - s0_b)) >> 23);
                    let scaled1 = interp1.wrapping_mul(vol) >> 7;
                    frac = frac.wrapping_add(step);
                    pos += (frac >> RARE_FRAC_BITS) as usize;
                    frac &= RARE_FRAC_MASK;

                    let packed = (scaled0 as u32).wrapping_add((scaled1 as u32) << 16);
                    dst[0] = (packed & 0xFFFF) as i16;
                    dst[1] = (packed >> 16) as i16;
                }

                voice.pos = pos;
                voice.frac = frac;
                return;
            }
        }

        // Standard per-sample `strh`/`ldrsh` path (all Standard variants, Banjo-Pilot slow path,
        // and Banjo-Pilot 4x-unrolled `ror #16` accumulate path which has zero cross-halfword bleed).
        if is_first {
            Self::mix_standard_impl::<true>(sample, voice, acc);
        } else {
            Self::mix_standard_impl::<false>(sample, voice, acc);
        }
    }

    /// Mix a single voice using [`RareAudioVersion::StandardSingleClamp`].
    #[inline]
    pub fn mix_voice_view(
        sample: &RareSampleView<'_>,
        voice: &mut RareVoice,
        acc: &mut [i16],
        is_first: bool,
    ) {
        Self::mix_voice_view_version(
            sample,
            voice,
            acc,
            is_first,
            RareAudioVersion::StandardSingleClamp,
        );
    }

    /// Clamp a 16-bit signed scratch accumulator into signed 8-bit DirectSound A FIFO samples
    /// (`[-128, +127]`), matching all three Rareware ARM clamp kernels (`SingleByteLdrsh`,
    /// `DualHalfCmn`, and `DualByteLdr32`).
    #[inline]
    pub fn clamp_i16_to_dma8(acc: &[i16], dma_out: &mut [i8]) {
        let n = acc.len().min(dma_out.len());
        for (&a, d) in acc[..n].iter().zip(dma_out[..n].iter_mut()) {
            *d = a.clamp(-128, 127) as i8;
        }
    }

    /// Render a complete DirectSound A mono DMA frame from all active voices in `self.voices`
    /// (`alloc` feature).
    #[cfg(feature = "alloc")]
    pub fn render_frame_dma8(&mut self, samples: &[RareSampleView<'_>], dma_out: &mut [i8]) {
        let mut acc = alloc::vec![0i16; dma_out.len()];
        let mut is_first = true;
        for voice in &mut self.voices {
            if !voice.active {
                continue;
            }
            if let Some(sample) = samples.get(voice.sample_index) {
                Self::mix_voice_view_version(sample, voice, &mut acc, is_first, self.version);
                is_first = false;
            }
        }
        Self::clamp_i16_to_dma8(&acc, dma_out);
    }
}

/// High-fidelity 48,000 Hz `f32` stereo software synthesizer for Rareware GBA samples.
///
/// Replaces the GBA's 13,379 Hz mono 8-bit linear-interpolated DirectSound A pipeline with
/// a compile-time normalized 1,024-phase 6-tap Lanczos-3 windowed-sinc polyphase resampler
/// and true equal-power stereo panning (`sqrt((1 ± pan) / 2)`).
#[derive(Debug, Clone, PartialEq)]
pub struct StudioSynth {
    /// Output sample rate in Hz (default [`Self::SAMPLE_RATE`] = `48,000` Hz).
    pub output_rate: u32,
    /// Native Rareware mixer reference rate used to interpret `RareVoice::step_q23` (default `13,379` Hz).
    pub source_mix_rate: u32,
    /// Active voices managed by this synthesizer instance (`alloc` feature).
    #[cfg(feature = "alloc")]
    pub voices: Vec<RareVoice>,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self::new(Self::SAMPLE_RATE)
    }
}

impl StudioSynth {
    /// Standard studio output sample rate (`48,000` Hz).
    pub const SAMPLE_RATE: u32 = 48_000;

    /// Create a new `StudioSynth` at `output_rate` Hz (typically `48,000` Hz).
    #[must_use]
    pub fn new(output_rate: u32) -> Self {
        Self {
            output_rate,
            source_mix_rate: RARE_DEFAULT_MIX_RATE,
            #[cfg(feature = "alloc")]
            voices: Vec::new(),
        }
    }

    /// Compute equal-power stereo gains `(gain_l, gain_r)` for `pan` in `[-1.0, +1.0]`.
    #[inline]
    #[must_use]
    pub fn equal_power_gains(pan: f32) -> (f32, f32) {
        let p = if pan.is_finite() {
            (pan as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let gain_l = sqrt_f64(0.5 * (1.0 - p)) as f32;
        let gain_r = sqrt_f64(0.5 * (1.0 + p)) as f32;
        (gain_l, gain_r)
    }

    /// Evaluate the 6-tap Lanczos-3 windowed-sinc interpolation at integer index `pos` and
    /// fractional offset `frac` in `[0.0, 1.0)`.
    #[inline]
    #[must_use]
    pub fn sample_lanczos3(sample: &RareSampleView<'_>, pos: usize, frac: f64) -> f32 {
        let frac = if frac.is_finite() {
            frac.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let phase_idx = ((frac * (LANCZOS_PHASES as f64) + 0.5) as usize).min(LANCZOS_PHASES);
        let w = &LANCZOS3_LUT[phase_idx];
        let effective_len = sample.sample_len.min(sample.pcm.len());
        if effective_len == 0 {
            return 0.0;
        }

        // Fast interior path: all 6 taps in bounds with zero boundary branches
        if pos >= 2 && pos + 3 < effective_len {
            let win = &sample.pcm[pos - 2..pos + 4];
            let s0 = (win[0] as i8) as f32;
            let s1 = (win[1] as i8) as f32;
            let s2 = (win[2] as i8) as f32;
            let s3 = (win[3] as i8) as f32;
            let s4 = (win[4] as i8) as f32;
            let s5 = (win[5] as i8) as f32;
            let val =
                (s0 * w[0] + s1 * w[1]) + (s2 * w[2] + s3 * w[3]) + (s4 * w[4] + s5 * w[5]);
            return val * (1.0 / 128.0);
        }

        let len = effective_len as isize;
        let loop_len = sample.loop_length as isize;
        let is_looped = sample.is_looped() && loop_len > 0 && loop_len <= len;
        let loop_start = if is_looped { len - loop_len } else { 0 };

        let mut acc = 0.0f32;
        for (tap, &wt) in w.iter().enumerate() {
            let mut idx = (pos as isize) + (tap as isize) - 2;
            if idx < 0 {
                idx = 0;
            } else if idx >= len {
                if is_looped {
                    idx = loop_start + ((idx - loop_start) % loop_len);
                } else {
                    idx = len - 1;
                }
            }
            let s = (sample.pcm[idx as usize] as i8) as f32;
            acc += s * wt;
        }
        acc * (1.0 / 128.0)
    }

    /// Render a single voice into separate left/right `f32` buffers using band-limited
    /// 6-tap Lanczos-3 interpolation and equal-power stereo panning.
    pub fn mix_voice_view_stereo(
        &self,
        sample: &RareSampleView<'_>,
        voice: &mut RareVoice,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        Self::mix_voice_view_stereo_at_rates(
            self.output_rate,
            self.source_mix_rate,
            sample,
            voice,
            out_l,
            out_r,
        );
    }

    /// Render a single voice into separate left/right `f32` buffers at explicit output/source rates.
    pub fn mix_voice_view_stereo_at_rates(
        output_rate: u32,
        source_mix_rate: u32,
        sample: &RareSampleView<'_>,
        voice: &mut RareVoice,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        if !voice.active || sample.is_empty() || output_rate == 0 {
            return;
        }
        let n = out_l.len().min(out_r.len());
        if n == 0 {
            return;
        }
        let pcm = sample.pcm;
        let sample_len = sample.sample_len.min(pcm.len());
        if sample_len == 0 {
            return;
        }

        // Convert Q9.23 step at source_mix_rate to floating-point step at output_rate
        let step_at_mix = (voice.step_q23 as f64) / ((1u64 << RARE_FRAC_BITS) as f64);
        let step_out = step_at_mix * (source_mix_rate as f64) / (output_rate as f64);

        // Bake PCM `1/128` normalization directly into stereo gains once per voice
        let vol_gain = (voice.volume as f32) * (1.0 / (128.0 * 128.0));
        let (pan_l, pan_r) = Self::equal_power_gains(voice.pan);
        let gain_l = vol_gain * pan_l;
        let gain_r = vol_gain * pan_r;

        let loop_len = sample.loop_length as usize;
        let is_looped = voice.playback_mode.is_looped() && loop_len > 0 && loop_len <= sample_len;
        let loop_start = sample_len.saturating_sub(loop_len);
        let sample_len_f = sample_len as f64;
        let loop_len_f = loop_len as f64;
        let safe_max_base = sample_len.saturating_sub(4);

        let out_l = &mut out_l[..n];
        let out_r = &mut out_r[..n];

        let mut pos_f =
            (voice.pos as f64) + ((voice.frac as f64) / ((1u64 << RARE_FRAC_BITS) as f64));
        let mut idx = 0usize;

        while idx < n {
            if pos_f >= sample_len_f {
                if is_looped {
                    pos_f -= loop_len_f;
                    if pos_f >= sample_len_f {
                        let loop_start_f = loop_start as f64;
                        pos_f = loop_start_f + ((pos_f - loop_start_f) % loop_len_f);
                    }
                } else {
                    voice.active = false;
                    break;
                }
            }

            let base = pos_f as usize;
            // Interior Q32.32 fixed-point fast path: zero boundary checks, zero modulo,
            // direct 6-tap slice loads, and balanced FMA reduction tree.
            if sample_len >= 6
                && base >= 2
                && base <= safe_max_base
                && step_out > 0.0
                && step_out < (1u64 << 31) as f64
            {
                let safe_limit = (safe_max_base as f64) + 0.999_999;
                let max_interior = (((safe_limit - pos_f) / step_out) as usize) + 1;
                let run = max_interior.min(n - idx);
                if run >= 4 {
                    const Q32_SCALE: f64 = 4_294_967_296.0;
                    let ph_q32 = (pos_f * Q32_SCALE) as u64;
                    let step_q32 = (step_out * Q32_SCALE) as u64;
                    let last_base = ((ph_q32
                        .saturating_add(step_q32.saturating_mul((run - 1) as u64)))
                        >> 32) as usize;
                    if last_base <= safe_max_base {
                        let src = &pcm[..last_base + 4];
                        let mut ph_shifted = ph_q32 - (2u64 << 32);
                        for (ol, or) in out_l[idx..idx + run]
                            .iter_mut()
                            .zip(&mut out_r[idx..idx + run])
                        {
                            let b0 = (ph_shifted >> 32) as usize;
                            // Nearest 1,024-phase index in `0..=1024`
                            let p = (((ph_shifted & 0xFFFF_FFFF) + (1 << 21)) >> 22) as usize;
                            let w = &LANCZOS3_LUT[p];
                            let win = &src[b0..b0 + 6];
                            let s0 = (win[0] as i8) as f32;
                            let s1 = (win[1] as i8) as f32;
                            let s2 = (win[2] as i8) as f32;
                            let s3 = (win[3] as i8) as f32;
                            let s4 = (win[4] as i8) as f32;
                            let s5 = (win[5] as i8) as f32;
                            // Balanced reduction tree halves FMA dependency chain latency
                            let val = (s0 * w[0] + s1 * w[1])
                                + (s2 * w[2] + s3 * w[3])
                                + (s4 * w[4] + s5 * w[5]);
                            *ol += val * gain_l;
                            *or += val * gain_r;
                            ph_shifted += step_q32;
                        }
                        pos_f += (run as f64) * step_out;
                        idx += run;
                        continue;
                    }
                }
            }

            // Boundary / loop-edge single-sample step
            if base >= sample_len {
                voice.active = false;
                break;
            }
            let frac = pos_f - (base as f64);
            let p = ((frac * (LANCZOS_PHASES as f64) + 0.5) as usize).min(LANCZOS_PHASES);
            let w = &LANCZOS3_LUT[p];
            let len_i = sample_len as isize;
            let loop_len_i = loop_len as isize;
            let loop_start_i = loop_start as isize;

            let mut val = 0.0f32;
            for (tap, &wt) in w.iter().enumerate() {
                let mut s_idx = (base as isize) + (tap as isize) - 2;
                if s_idx < 0 {
                    s_idx = 0;
                } else if s_idx >= len_i {
                    if is_looped {
                        s_idx = loop_start_i + ((s_idx - loop_start_i) % loop_len_i);
                    } else {
                        s_idx = len_i - 1;
                    }
                }
                val += ((pcm[s_idx as usize] as i8) as f32) * wt;
            }

            out_l[idx] += val * gain_l;
            out_r[idx] += val * gain_r;
            pos_f += step_out;
            idx += 1;
        }

        if voice.active {
            if pos_f >= sample_len_f {
                if is_looped {
                    pos_f -= loop_len_f;
                    if pos_f >= sample_len_f {
                        let loop_start_f = loop_start as f64;
                        pos_f = loop_start_f + ((pos_f - loop_start_f) % loop_len_f);
                    }
                } else {
                    voice.active = false;
                    return;
                }
            }
            let pos_i = pos_f as usize;
            let frac = pos_f - (pos_i as f64);
            voice.pos = pos_i;
            voice.frac = ((frac * ((1u64 << RARE_FRAC_BITS) as f64)) as u32) & RARE_FRAC_MASK;
        }
    }

    /// Render all active voices in `self.voices` into interleaved stereo `f32` `[L, R, L, R, ...]`
    /// (`alloc` feature).
    #[cfg(feature = "alloc")]
    pub fn render_interleaved_stereo(
        &mut self,
        samples: &[RareSampleView<'_>],
        interleaved_out: &mut [f32],
    ) {
        let frames = interleaved_out.len() / 2;
        let mut l = alloc::vec![0.0f32; frames];
        let mut r = alloc::vec![0.0f32; frames];
        let output_rate = self.output_rate;
        let source_mix_rate = self.source_mix_rate;
        for voice in &mut self.voices {
            if !voice.active {
                continue;
            }
            if let Some(sample) = samples.get(voice.sample_index) {
                Self::mix_voice_view_stereo_at_rates(
                    output_rate,
                    source_mix_rate,
                    sample,
                    voice,
                    &mut l,
                    &mut r,
                );
            }
        }
        for (pair, (&sl, &sr)) in interleaved_out
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(l.iter().zip(r.iter()))
        {
            pair[0] = sl;
            pair[1] = sr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::RarePlaybackMode;

    #[test]
    fn test_studio_synth_1024_lut_matches_direct_lanczos3() {
        let mut pcm = [0u8; 65];
        for (i, slot) in pcm.iter_mut().enumerate() {
            let v = (const_sin((i as f64) * 0.35) * 100.0) as i8;
            *slot = v as u8;
        }
        let view = RareSampleView::from_signed_pcm(
            0,
            RARE_DEFAULT_MIX_RATE,
            60,
            RarePlaybackMode::MidiOneShot,
            0,
            &pcm,
            true,
        );
        let synth = StudioSynth::new(StudioSynth::SAMPLE_RATE);
        let mut voice = RareVoice::with_phase(
            0,
            RarePlaybackMode::MidiOneShot,
            5,
            0x0020_0000, // 0.25
            RARE_FRAC_ONE,
            128,
        );
        let mut out_l = [0.0f32; 128];
        let mut out_r = [0.0f32; 128];
        synth.mix_voice_view_stereo(&view, &mut voice, &mut out_l, &mut out_r);

        let step = (RARE_DEFAULT_MIX_RATE as f64) / (StudioSynth::SAMPLE_RATE as f64);
        let (gain_l, _) = StudioSynth::equal_power_gains(0.0);
        let mut ref_pos = 5.25f64;
        for &actual in &out_l {
            let center = ref_pos as isize;
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;
            for k in (center - 2)..=(center + 3) {
                let s = (pcm[k as usize] as i8) as f64 / 128.0;
                let w = lanczos_kernel(ref_pos - (k as f64), 3.0);
                acc += s * w;
                wsum += w;
            }
            let expected = ((acc / wsum) as f32) * gain_l;
            assert!(
                (actual - expected).abs() < 1e-3,
                "LUT mismatch: actual={actual}, expected={expected}"
            );
            ref_pos += step;
        }
    }

    #[test]
    fn test_packed_fast_path_1to1_preserves_negative_bleed() {
        let pcm: [u8; 9] = [
            (-10i8) as u8,
            20u8,
            10u8,
            20u8,
            (-1i8) as u8,
            0u8,
            0u8,
            (-5i8) as u8,
            0u8,
        ];
        let view = RareSampleView::from_signed_pcm(
            0,
            RARE_DEFAULT_MIX_RATE,
            60,
            RarePlaybackMode::MidiOneShot,
            0,
            &pcm,
            true,
        );
        let mut voice = RareVoice::with_phase(
            0,
            RarePlaybackMode::MidiOneShot,
            0,
            0,
            RARE_FRAC_ONE,
            128,
        );
        let mut acc = [0i16; 6];
        AccuracySynth::mix_voice_view_version(
            &view,
            &mut voice,
            &mut acc,
            true,
            RareAudioVersion::PackedFastPath,
        );
        // Pair 0: (-10, 20) -> -10 < 0 bleeds -1 into 20 => (-10, 19)
        // Pair 1: (10, 20)  -> 10 >= 0, no bleed => (10, 20)
        // Pair 2: (-1, 0)   -> -1 < 0 bleeds -1 into 0 => (-1, -1)
        assert_eq!(acc, [-10, 19, 10, 20, -1, -1]);
        assert_eq!(voice.pos, 6);
        assert_eq!(voice.frac, 0);
    }
}
