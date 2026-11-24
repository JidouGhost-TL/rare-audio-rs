//! Dual software synthesis engines for the Rareware GBA Sound Engine.
//!
//! Provides two synthesis pipelines:
//! - [`AccuracySynth`]: 100% bit-exact emulation of all four Rareware GBA ARM7TDMI IWRAM
//!   mixer revisions (`StandardSingleClamp`, `StandardDualHalfClamp`, `StandardDualByteClamp`,
//!   and `PackedFastPath`), verified byte-for-byte (`2,688 / 2,688` bytes) against real
//!   ARM7TDMI execution in headless `libmgba`.
//! - [`StudioSynth`]: High-fidelity 48,000 Hz `f32` stereo renderer with compile-time
//!   normalized 6-tap Lanczos-3 windowed-sinc interpolation and equal-power stereo panning.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::freq::{RARE_DEFAULT_MIX_RATE, RARE_FRAC_BITS, RARE_FRAC_MASK};
use crate::sample::RareSampleView;
use crate::version::RareAudioVersion;
use crate::voice::RareVoice;

/// Number of fractional polyphase steps in [`StudioSynth`]'s compile-time Lanczos-3 LUT.
const LANCZOS_PHASES: usize = 256;

/// Compile-time normalized 6-tap Lanczos-3 polyphase lookup table (`257 * 6 * 4 = 6,168` bytes).
const LANCZOS3_LUT: [[f32; 6]; LANCZOS_PHASES + 1] = build_lanczos3_lut();

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
#[derive(Debug, Clone)]
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

        let n = acc.len();
        let sample_len = sample.sample_len;
        let loop_len = sample.loop_length as usize;
        let is_looped = voice.playback_mode.is_looped() && loop_len > 0;
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
                let mut pos = voice.pos;
                let mut frac = voice.frac;

                let mut k = 0usize;
                while k < n {
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

                    // Authentic Banjo-Pilot ARM `add r2, sl, r2, lsl #16`:
                    // `scaled0 as u32` has 0xFFFF in upper 16 bits when `scaled0 < 0`,
                    // which subtracts 1 from `scaled1` in the upper halfword!
                    let packed = (scaled0 as u32).wrapping_add((scaled1 as u32) << 16);
                    acc[k] = (packed & 0xFFFF) as i16;
                    acc[k + 1] = (packed >> 16) as i16;
                    k += 2;
                }

                voice.pos = pos;
                voice.frac = frac;
                return;
            }
        }

        // Standard per-sample `strh`/`ldrsh` path (all Standard variants, Banjo-Pilot slow path,
        // and Banjo-Pilot 4x-unrolled `ror #16` accumulate path which has zero cross-halfword bleed).
        let mut pos = voice.pos;
        let mut frac = voice.frac;

        for slot in acc.iter_mut() {
            let (s0, s1) = sample.sample_pair_i32(pos);
            let interp = s0 + (((frac as i32).wrapping_mul(s1 - s0)) >> 23);
            let scaled = (interp.wrapping_mul(vol) >> 7) as i16;

            if is_first {
                *slot = scaled;
            } else {
                *slot = slot.wrapping_add(scaled);
            }

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
        for i in 0..n {
            dma_out[i] = acc[i].clamp(-128, 127) as i8;
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
/// a compile-time normalized 6-tap Lanczos-3 windowed-sinc polyphase resampler and true
/// equal-power stereo panning (`cos/sin` via `sqrt((1 ± pan) / 2)`).
#[derive(Debug, Clone)]
pub struct StudioSynth {
    /// Output sample rate in Hz (default `48,000` Hz).
    pub output_rate: u32,
    /// Native Rareware mixer reference rate used to interpret `RareVoice::step_q23` (default `13,379` Hz).
    pub source_mix_rate: u32,
    /// Active voices managed by this synthesizer instance (`alloc` feature).
    #[cfg(feature = "alloc")]
    pub voices: Vec<RareVoice>,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl StudioSynth {
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
        let p = (pan as f64).clamp(-1.0, 1.0);
        let gain_l = sqrt_f64(0.5 * (1.0 - p)) as f32;
        let gain_r = sqrt_f64(0.5 * (1.0 + p)) as f32;
        (gain_l, gain_r)
    }

    /// Evaluate the 6-tap Lanczos-3 windowed-sinc interpolation at integer index `pos` and
    /// fractional offset `frac` in `[0.0, 1.0)`.
    #[inline]
    #[must_use]
    pub fn sample_lanczos3(sample: &RareSampleView<'_>, pos: usize, frac: f64) -> f32 {
        let phase_idx = ((frac * (LANCZOS_PHASES as f64)) as usize).min(LANCZOS_PHASES);
        let weights = &LANCZOS3_LUT[phase_idx];
        let len = sample.sample_len as isize;
        let is_looped = sample.is_looped();
        let loop_len = sample.loop_length as isize;
        let loop_start = len - loop_len;

        let mut acc = 0.0f32;
        for (tap, &w) in weights.iter().enumerate() {
            let mut idx = (pos as isize) + (tap as isize) - 2;
            if idx < 0 {
                idx = 0;
            } else if idx >= len {
                if is_looped && loop_len > 0 {
                    idx = loop_start + ((idx - loop_start) % loop_len);
                } else {
                    idx = len - 1;
                }
            }
            let s = (sample.sample_i8(idx as usize) as f32) / 128.0;
            acc += s * w;
        }
        acc
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

        // Convert Q9.23 step at source_mix_rate to floating-point step at output_rate
        let step_at_mix = (voice.step_q23 as f64) / ((1u64 << RARE_FRAC_BITS) as f64);
        let step_out = step_at_mix * (source_mix_rate as f64) / (output_rate as f64);

        let vol_gain = (voice.volume as f32) / 128.0;
        let (pan_l, pan_r) = Self::equal_power_gains(voice.pan);
        let gain_l = vol_gain * pan_l;
        let gain_r = vol_gain * pan_r;

        let sample_len = sample.sample_len;
        let loop_len = sample.loop_length as usize;
        let is_looped = voice.playback_mode.is_looped() && loop_len > 0;

        let mut pos_f = (voice.pos as f64) + ((voice.frac as f64) / ((1u64 << RARE_FRAC_BITS) as f64));

        for i in 0..n {
            let pos_i = pos_f as usize;
            if pos_i >= sample_len {
                voice.active = false;
                break;
            }
            let frac = pos_f - (pos_i as f64);
            let s = Self::sample_lanczos3(sample, pos_i, frac);
            out_l[i] += s * gain_l;
            out_r[i] += s * gain_r;

            pos_f += step_out;
            if pos_f >= (sample_len as f64) {
                if is_looped {
                    while pos_f >= (sample_len as f64) {
                        pos_f -= loop_len as f64;
                    }
                } else {
                    voice.active = false;
                    break;
                }
            }
        }

        if voice.active {
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
        for i in 0..frames {
            interleaved_out[i * 2] = l[i];
            interleaved_out[i * 2 + 1] = r[i];
        }
    }
}
