//! Hardware timer, Q9.23 fixed-point phase stepping, and volume scaling math
//! for the Rareware GBA Sound Engine.

/// GBA ARM7TDMI master clock frequency (`16,777,216` Hz = `2^24` Hz).
pub const GBA_CLOCK_HZ: u32 = 16_777_216;

/// Default Rareware GBA DirectSound A master mixing rate (`13,379` Hz, `0x3443`).
pub const RARE_DEFAULT_MIX_RATE: u32 = 13_379;

/// All four hardware DirectSound A mixing rates present in Rareware GBA driver tables.
pub const RARE_SUPPORTED_MIX_RATES: [u32; 4] = [10_512, 13_379, 18_157, 21_024];

/// Default GBA Timer 0 reload value (`0xFB1A` -> `65536 - 0xFB1A = 1254`, `16,777,216 / 1254 ≈ 13,379` Hz).
pub const RARE_TIMER0_RELOAD: u16 = 0xFB1A;

/// Default DirectSound A DMA1 buffer length per video frame (`224` mono 8-bit samples = `0x38` words).
pub const RARE_DEFAULT_FRAME_SAMPLES: usize = 224;

/// Number of fractional bits in Rareware's Q9.23 phase accumulator (`23` bits).
pub const RARE_FRAC_BITS: u32 = 23;

/// Bitmask extracting the 23-bit fractional phase (`0x007F_FFFF`).
pub const RARE_FRAC_MASK: u32 = (1u32 << RARE_FRAC_BITS) - 1;

/// Fixed-point 1.0 in Rareware's Q9.23 phase accumulator (`0x0080_0000`).
pub const RARE_FRAC_ONE: u32 = 1u32 << RARE_FRAC_BITS;

/// Compute the hardware DirectSound A sample rate in Hz from a GBA Timer 0 reload value.
#[inline]
#[must_use]
pub const fn rate_from_timer_reload(reload: u16) -> u32 {
    let ticks = 65_536u32 - (reload as u32);
    if ticks == 0 {
        return 0;
    }
    (GBA_CLOCK_HZ + (ticks / 2)) / ticks
}

/// Convert a source sample frequency (Hz) to a Q9.23 phase step at [`RARE_DEFAULT_MIX_RATE`].
#[inline]
#[must_use]
pub const fn freq_to_step_q23(freq_hz: u32) -> u32 {
    freq_to_step_q23_at_rate(freq_hz, RARE_DEFAULT_MIX_RATE)
}

/// Convert a source sample frequency (Hz) to a Q9.23 phase step at an arbitrary mixer rate (Hz).
#[inline]
#[must_use]
pub const fn freq_to_step_q23_at_rate(freq_hz: u32, mix_rate_hz: u32) -> u32 {
    if mix_rate_hz == 0 {
        return 0;
    }
    let num = (freq_hz as u64) << RARE_FRAC_BITS;
    (num / (mix_rate_hz as u64)) as u32
}

/// Convert a Q9.23 phase step back to playback frequency in Hz at `mix_rate_hz`.
#[inline]
#[must_use]
pub const fn step_q23_to_freq(step_q23: u32, mix_rate_hz: u32) -> u32 {
    (((step_q23 as u64) * (mix_rate_hz as u64)) >> RARE_FRAC_BITS) as u32
}

/// Multiply a Q9.23 base step by a Q16.16 pitch multiplier, matching Rareware's exact ARM
/// `umull sb, sl, r6, r7; lsr r7, sb, #16; add r7, r7, sl, lsl #16` instruction sequence.
#[inline]
#[must_use]
pub const fn apply_pitch_scale_q16(base_step_q23: u32, pitch_scale_q16: u32) -> u32 {
    let prod = (base_step_q23 as u64) * (pitch_scale_q16 as u64);
    (prod >> 16) as u32
}

/// Compute Rareware's effective voice volume (`r4`) matching the exact ARM shift chain:
/// - For MIDI voices (`midi_ch_vol: Some(v)`): `vol = (master_vol * (v + 1)) >> 7`
/// - `vol = (vol * voice_vol) >> 7`
/// - `vol = (vol * envelope_level) >> 19`
///
/// The resulting `vol` is then used directly in the inner loop as `(interp * vol) >> 7`.
#[inline]
#[must_use]
pub const fn compute_effective_volume(
    master_vol: u32,
    midi_ch_vol: Option<u8>,
    voice_vol: u8,
    envelope_level: u32,
) -> i32 {
    let mut r4 = master_vol;
    if let Some(ch_v) = midi_ch_vol {
        r4 = r4.wrapping_mul((ch_v as u32) + 1) >> 7;
    }
    r4 = r4.wrapping_mul(voice_vol as u32) >> 7;
    r4 = r4.wrapping_mul(envelope_level) >> 19;
    r4 as i32
}

/// 12-TET semitone ratio table (`2^(k/12)` scaled to `16.16` fixed-point for `k = 0..=12`).
const SEMITONE_RATIOS_Q16: [u32; 13] = [
    65_536,  // 0: 1.000000
    69_433,  // 1: 2^(1/12)
    73_562,  // 2: 2^(2/12)
    77_936,  // 3: 2^(3/12)
    82_570,  // 4: 2^(4/12)
    87_480,  // 5: 2^(5/12)
    92_682,  // 6: 2^(6/12)
    98_193,  // 7: 2^(7/12)
    104_032, // 8: 2^(8/12)
    110_218, // 9: 2^(9/12)
    116_772, // 10: 2^(10/12)
    123_715, // 11: 2^(11/12)
    131_072, // 12: 2.000000
];

/// Compute a Q9.23 phase step for playing a sample at a target MIDI note relative to its root note.
#[must_use]
pub const fn note_to_step_q23(
    sample_rate_hz: u32,
    root_note: u8,
    target_note: u8,
    mix_rate_hz: u32,
) -> u32 {
    let base = freq_to_step_q23_at_rate(sample_rate_hz, mix_rate_hz) as u64;
    let diff = (target_note as i32) - (root_note as i32);
    let octaves = diff.div_euclid(12);
    let semi = diff.rem_euclid(12) as usize;
    let scaled = (base * (SEMITONE_RATIOS_Q16[semi] as u64)) >> 16;
    if octaves >= 0 {
        let shift = if octaves > 16 { 16 } else { octaves as u32 };
        (scaled << shift) as u32
    } else {
        let shift = if -octaves > 31 { 31 } else { (-octaves) as u32 };
        (scaled >> shift) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer0_reload_matches_13379_hz() {
        assert_eq!(rate_from_timer_reload(RARE_TIMER0_RELOAD), 13_379);
    }

    #[test]
    fn test_freq_to_step_q23_identity() {
        assert_eq!(freq_to_step_q23(RARE_DEFAULT_MIX_RATE), RARE_FRAC_ONE);
        assert_eq!(
            step_q23_to_freq(RARE_FRAC_ONE, RARE_DEFAULT_MIX_RATE),
            RARE_DEFAULT_MIX_RATE
        );
    }

    #[test]
    fn test_note_to_step_q23_octaves() {
        let c4 = note_to_step_q23(13_379, 60, 60, 13_379);
        let c5 = note_to_step_q23(13_379, 60, 72, 13_379);
        let c3 = note_to_step_q23(13_379, 60, 48, 13_379);
        assert_eq!(c4, RARE_FRAC_ONE);
        assert_eq!(c5, RARE_FRAC_ONE * 2);
        assert_eq!(c3, RARE_FRAC_ONE / 2);
    }

    #[test]
    fn test_apply_pitch_scale_q16() {
        assert_eq!(apply_pitch_scale_q16(RARE_FRAC_ONE, 0x0001_0000), RARE_FRAC_ONE);
        assert_eq!(
            apply_pitch_scale_q16(RARE_FRAC_ONE, 0x0002_0000),
            RARE_FRAC_ONE * 2
        );
    }

    #[test]
    fn test_compute_effective_volume() {
        // Full master (128), full ch_vol (127 -> 128), full voice (128), full env (1 << 19)
        let vol = compute_effective_volume(128, Some(127), 128, 1 << 19);
        assert_eq!(vol, 128);
    }
}
