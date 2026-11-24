//! Live voice channel state matching the Rareware GBA Sound Engine's 40-byte (`0x28`) voice slot.

use crate::freq::{freq_to_step_q23_at_rate, RARE_DEFAULT_MIX_RATE, RARE_FRAC_MASK};
use crate::sample::RarePlaybackMode;

/// Voice state flag written by the ARM mixer when a one-shot sample finishes (`0x10`).
pub const RARE_VOICE_STATE_STOPPED: u8 = 0x10;

/// Active voice state flag for a mixing channel (`0x12` in retail builds).
pub const RARE_VOICE_STATE_ACTIVE: u8 = 0x12;

/// Live Rareware GBA mixer voice slot.
///
/// Tracks the exact integer byte position (`pos`), 23-bit fractional phase (`frac`),
/// Q9.23 phase increment (`step_q23`), effective mixer volume (`volume`, where `128` = 1.0x),
/// and optional stereo pan (`-1.0..=1.0`, used by [`crate::StudioSynth`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RareVoice {
    /// Index into the sample bank.
    pub sample_index: usize,
    /// Playback mode (`1..=4`: MIDI/SFX, one-shot/looped).
    pub playback_mode: RarePlaybackMode,
    /// Current integer sample index in PCM data.
    pub pos: usize,
    /// Current 23-bit fractional phase (`0..=0x007F_FFFF`).
    pub frac: u32,
    /// Q9.23 fixed-point phase increment per output sample (`0x0080_0000` = 1:1).
    pub step_q23: u32,
    /// Effective hardware volume multiplier (`r4`, scaled by `>> 7` in inner loop; `128` = 1.0x).
    pub volume: i32,
    /// Stereo pan for [`crate::StudioSynth`] (`-1.0` = hard left, `0.0` = center, `+1.0` = hard right).
    pub pan: f32,
    /// Whether the voice is currently active.
    pub active: bool,
}

impl RareVoice {
    /// Create a new active voice from a playback frequency (Hz) at [`RARE_DEFAULT_MIX_RATE`].
    #[must_use]
    pub const fn new(
        sample_index: usize,
        playback_mode: RarePlaybackMode,
        freq_hz: u32,
        volume: i32,
    ) -> Self {
        Self::with_step(
            sample_index,
            playback_mode,
            freq_to_step_q23_at_rate(freq_hz, RARE_DEFAULT_MIX_RATE),
            volume,
        )
    }

    /// Create a new active voice with an explicit Q9.23 phase step.
    #[must_use]
    pub const fn with_step(
        sample_index: usize,
        playback_mode: RarePlaybackMode,
        step_q23: u32,
        volume: i32,
    ) -> Self {
        Self {
            sample_index,
            playback_mode,
            pos: 0,
            frac: 0,
            step_q23,
            volume,
            pan: 0.0,
            active: true,
        }
    }

    /// Create a new active voice with explicit initial position, 23-bit fractional phase,
    /// Q9.23 step, and effective volume.
    #[must_use]
    pub const fn with_phase(
        sample_index: usize,
        playback_mode: RarePlaybackMode,
        pos: usize,
        frac: u32,
        step_q23: u32,
        volume: i32,
    ) -> Self {
        Self {
            sample_index,
            playback_mode,
            pos,
            frac: frac & RARE_FRAC_MASK,
            step_q23,
            volume,
            pan: 0.0,
            active: true,
        }
    }
}
