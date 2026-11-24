//! # rare-audio
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact golden-tested
//! emulator for the **Rareware GBA Sound Engine** on the Game Boy Advance.
//!
//! The Rareware GBA Sound Engine powered Rare's entire Game Boy Advance catalog including
//! *Donkey Kong Country 1–3*, *Banjo-Kazooie: Grunty's Revenge*, *Banjo-Pilot*,
//! *Sabre Wulf*, *It's Mr. Pants*, and prototypes (*Battletoads*, *Diddy Kong Pilot*).
//!
//! ## Features
//! - **Zero dependencies** in `[dependencies]`, `#![no_std]` compatible, `#![deny(unsafe_code)]`.
//! - **Native Rareware Binary Parsers**: Zero-allocation [`RareSampleDesc`] (68-byte descriptor),
//!   [`RareSongHeader`] (20-byte song header), and [`RareSampleView`] (signed 8-bit PCM + 1-byte
//!   interpolation guard).
//! - **Dual Synthesis Pipelines**:
//!   - [`AccuracySynth`]: Bit-exact emulation of all 4 Rareware GBA ARM7TDMI IWRAM mixer
//!     revisions ([`RareAudioVersion::StandardSingleClamp`], [`RareAudioVersion::StandardDualHalfClamp`],
//!     [`RareAudioVersion::StandardDualByteClamp`], and [`RareAudioVersion::PackedFastPath`]
//!     including *Banjo-Pilot*'s authentic ARM negative-low-halfword `-1` carry bleed).
//!   - [`StudioSynth`]: 48,000 Hz `f32` high-fidelity stereo renderer with compile-time
//!     6-tap Lanczos-3 windowed-sinc interpolation and equal-power stereo panning.
//! - **Byte-Exact mGBA ARM Golden Tested**: 2,688 / 2,688 output bytes verified strictly
//!   byte-for-byte against real ARM7TDMI execution in headless `libmgba` across all 4
//!   Rareware GBA mixer revisions.
//!
//! ## Quickstart
//!
//! ```rust
//! use rare_audio::{
//!     AccuracySynth, RareAudioVersion, RarePlaybackMode, RareSampleView, RareVoice,
//!     RARE_DEFAULT_FRAME_SAMPLES, RARE_DEFAULT_MIX_RATE,
//! };
//!
//! // Signed 8-bit PCM with 1-byte interpolation guard
//! let pcm_i8: &[u8] = &[0, 32, 64, 96, 64, 32, 0, 224, 192, 160, 192, 224, 0];
//! let sample = RareSampleView::from_signed_pcm(
//!     0,
//!     RARE_DEFAULT_MIX_RATE,
//!     60,
//!     RarePlaybackMode::MidiLooped,
//!     12,
//!     pcm_i8,
//!     true,
//! );
//!
//! let mut voice = RareVoice::new(0, RarePlaybackMode::MidiLooped, RARE_DEFAULT_MIX_RATE, 128);
//! let mut acc = [0i16; RARE_DEFAULT_FRAME_SAMPLES];
//! let mut dma = [0i8; RARE_DEFAULT_FRAME_SAMPLES];
//!
//! AccuracySynth::mix_voice_view_version(
//!     &sample,
//!     &mut voice,
//!     &mut acc,
//!     true,
//!     RareAudioVersion::StandardSingleClamp,
//! );
//! AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
//! assert_eq!(dma[0], 0);
//! assert_eq!(dma[3], 96);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod freq;
pub mod sample;
pub mod synth;
pub mod version;
pub mod voice;

pub use error::{Error, Result};
pub use freq::{
    apply_pitch_scale_q16, compute_effective_volume, freq_to_step_q23, freq_to_step_q23_at_rate,
    note_to_step_q23, rate_from_timer_reload, step_q23_to_freq, GBA_CLOCK_HZ,
    RARE_DEFAULT_FRAME_SAMPLES, RARE_DEFAULT_MIX_RATE, RARE_FRAC_BITS, RARE_FRAC_MASK,
    RARE_FRAC_ONE, RARE_SUPPORTED_MIX_RATES, RARE_TIMER0_RELOAD,
};
#[cfg(feature = "alloc")]
pub use sample::RareSample;
pub use sample::{
    RareAdsr, RarePlaybackMode, RareSampleDesc, RareSampleView, RareSongHeader, GBA_ROM_BUS_BASE,
    RARE_DESC_MAGIC_SIZE, RARE_SAMPLE_DESC_SIZE, RARE_SONG_HEADER_SIZE,
};
pub use synth::{sqrt_f64, AccuracySynth, StudioSynth};
pub use version::{
    RareAudioVersion, RareClampMode, RARE_DEBUG_SIGNATURE, RARE_FIFO_DMA_SIGNATURE,
};
pub use voice::{RareVoice, RARE_VOICE_STATE_ACTIVE, RARE_VOICE_STATE_STOPPED};
