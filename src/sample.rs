//! Rareware GBA Sound Engine sample descriptors, song headers, and zero-allocation sample views.
//!
//! Every melodic instrument, percussion hit, and sound effect in the Rareware GBA Sound Engine
//! uses a 68-byte (`0x44`) little-endian [`RareSampleDesc`] descriptor pointing to raw signed
//! 8-bit PCM (`i8`) in ROM, followed by a 1-byte linear-interpolation guard byte.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::error::{Error, Result};

/// GBA ROM bus window base address (`0x0800_0000`).
pub const GBA_ROM_BUS_BASE: u32 = 0x0800_0000;

/// Size of a Rareware `RareSampleDesc` binary descriptor in bytes (`0x44` = 68 bytes).
pub const RARE_SAMPLE_DESC_SIZE: usize = 68;

/// Expected `magic_size` field at `+0x00` of every valid Rareware sample descriptor (`0x20`).
pub const RARE_DESC_MAGIC_SIZE: u32 = 0x20;

/// Size of a Rareware `RareSongHeader` binary struct in bytes (`0x14` = 20 bytes).
pub const RARE_SONG_HEADER_SIZE: usize = 20;

/// Playback mode stored at `+0x04` of [`RareSampleDesc`] and `+0x01` of a live voice slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RarePlaybackMode {
    /// Mode 1: MIDI instrument one-shot sample.
    MidiOneShot = 1,
    /// Mode 2: MIDI instrument forward-looped sample (`pos -= loop_length` on `pos >= sample_end`).
    MidiLooped = 2,
    /// Mode 3: Sound effect one-shot sample.
    SfxOneShot = 3,
    /// Mode 4: Sound effect forward-looped sample (`pos -= loop_length` on `pos >= sample_end`).
    SfxLooped = 4,
}

impl RarePlaybackMode {
    /// Parse a raw `u32` playback mode (`1..=4`).
    pub const fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            1 => Ok(Self::MidiOneShot),
            2 => Ok(Self::MidiLooped),
            3 => Ok(Self::SfxOneShot),
            4 => Ok(Self::SfxLooped),
            other => Err(Error::InvalidPlaybackMode(other)),
        }
    }

    /// Returns `true` for looped playback modes (`MidiLooped` or `SfxLooped`).
    #[inline]
    #[must_use]
    pub const fn is_looped(self) -> bool {
        matches!(self, Self::MidiLooped | Self::SfxLooped)
    }

    /// Returns `true` for sound-effect playback modes (`SfxOneShot` or `SfxLooped`).
    #[inline]
    #[must_use]
    pub const fn is_sfx(self) -> bool {
        matches!(self, Self::SfxOneShot | Self::SfxLooped)
    }
}

/// ADSR envelope parameters (`0..=99`, where `99` = instant attack / maximum level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RareAdsr {
    /// Attack rate (`0..=99`; `99` = instant).
    pub attack: u8,
    /// Decay rate (`0..=99`).
    pub decay: u8,
    /// Sustain level (`0..=99`).
    pub sustain: u8,
    /// Release rate (`0..=99`).
    pub release: u8,
}

/// Parsed 68-byte (`0x44`) Rareware GBA sample descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RareSampleDesc {
    /// Playback mode (`1..=4`).
    pub playback_mode: RarePlaybackMode,
    /// Native playback sample rate in Hz (`4000..=48000`; e.g. `11025`, `13379`).
    pub sample_rate_hz: u32,
    /// MIDI root note (`0..=127`; `60` = Middle C / C4).
    pub root_note: u8,
    /// GBA ROM pointer (`0x08xxxxxx`) to the start of signed 8-bit PCM data.
    pub sample_start_ptr: u32,
    /// Loop length in bytes (`0..=sample_len`).
    pub loop_length: u32,
    /// GBA ROM pointer (`0x08xxxxxx`) to the exclusive end of signed 8-bit PCM data.
    pub sample_end_ptr: u32,
    /// ADSR envelope settings (`0..=99` per stage).
    pub adsr: RareAdsr,
    /// Signed fine-tune offset in cents.
    pub fine_tune_cents: i32,
    /// Pitch-bend range in semitones.
    pub pitch_bend_range: u32,
    /// LFO vibrato rate.
    pub vibrato_rate: u32,
    /// LFO vibrato depth.
    pub vibrato_depth: u32,
}

impl RareSampleDesc {
    /// Parse and validate a 68-byte (`0x44`) little-endian Rareware sample descriptor.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < RARE_SAMPLE_DESC_SIZE {
            return Err(Error::UnexpectedEof);
        }

        let magic_size = read_u32_le(data, 0x00);
        if magic_size != RARE_DESC_MAGIC_SIZE {
            return Err(Error::InvalidHeader("magic_size != 0x20"));
        }

        let playback_mode = RarePlaybackMode::from_raw(read_u32_le(data, 0x04))?;
        let sample_rate_hz = read_u32_le(data, 0x08);
        if !(4_000..=48_000).contains(&sample_rate_hz) {
            return Err(Error::InvalidHeader("sample_rate_hz out of 4000..=48000"));
        }

        let root_note = read_u32_le(data, 0x0C);
        if root_note > 127 {
            return Err(Error::InvalidHeader("root_note > 127"));
        }

        let sample_start_ptr = read_u32_le(data, 0x10);
        let loop_length = read_u32_le(data, 0x14);
        let sample_end_ptr = read_u32_le(data, 0x18);
        let reserved_0 = read_u32_le(data, 0x1C);
        let reserved_1 = read_u32_le(data, 0x20);

        if reserved_0 != 0 || reserved_1 != 0 {
            return Err(Error::InvalidHeader("reserved fields must be zero"));
        }
        if sample_start_ptr < GBA_ROM_BUS_BASE || sample_end_ptr <= sample_start_ptr {
            return Err(Error::InvalidHeader("invalid ROM sample pointer range"));
        }

        let sample_len = (sample_end_ptr - sample_start_ptr) as usize;
        if (loop_length as usize) > sample_len {
            return Err(Error::InvalidLoopRange {
                loop_length,
                pcm_length: sample_len,
            });
        }

        let att = read_u32_le(data, 0x24);
        let dec = read_u32_le(data, 0x28);
        let sus = read_u32_le(data, 0x2C);
        let rel = read_u32_le(data, 0x30);
        if att > 99 || dec > 99 || sus > 99 || rel > 99 {
            return Err(Error::InvalidHeader("ADSR stage > 99"));
        }

        let fine_tune_cents = read_u32_le(data, 0x34) as i32;
        let pitch_bend_range = read_u32_le(data, 0x38);
        let vibrato_rate = read_u32_le(data, 0x3C);
        let vibrato_depth = read_u32_le(data, 0x40);

        Ok(Self {
            playback_mode,
            sample_rate_hz,
            root_note: root_note as u8,
            sample_start_ptr,
            loop_length,
            sample_end_ptr,
            adsr: RareAdsr {
                attack: att as u8,
                decay: dec as u8,
                sustain: sus as u8,
                release: rel as u8,
            },
            fine_tune_cents,
            pitch_bend_range,
            vibrato_rate,
            vibrato_depth,
        })
    }

    /// Total PCM payload length in bytes (`sample_end_ptr - sample_start_ptr`).
    #[inline]
    #[must_use]
    pub const fn sample_len(&self) -> usize {
        (self.sample_end_ptr - self.sample_start_ptr) as usize
    }

    /// Returns `true` if this descriptor loops during playback.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.playback_mode.is_looped() && self.loop_length > 0
    }

    /// Loop start sample index (`sample_len - loop_length`), if looped.
    #[inline]
    #[must_use]
    pub const fn loop_start(&self) -> Option<usize> {
        if self.is_looped() {
            Some(self.sample_len() - (self.loop_length as usize))
        } else {
            None
        }
    }

    /// ROM file offset where the signed 8-bit PCM payload begins.
    #[inline]
    #[must_use]
    pub const fn rom_pcm_offset(&self) -> usize {
        (self.sample_start_ptr - GBA_ROM_BUS_BASE) as usize
    }
}

/// Parsed 20-byte (`0x14`) Rareware GBA song header (`RareSongHeader`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RareSongHeader {
    /// Number of MIDI tracks (typically `3..=16`).
    pub num_tracks: u32,
    /// Pulses per quarter note (typically `384` = `0x0180`).
    pub ticks_per_quarter: u32,
    /// GBA ROM pointer to `[u32; num_tracks]` MIDI track pointers.
    pub track_ptr_table: u32,
    /// GBA ROM pointer to `[u32; 128]` melodic [`RareSampleDesc`] pointers.
    pub instrument_table_ptr: u32,
    /// GBA ROM pointer to percussion [`RareSampleDesc`] pointers.
    pub percussion_table_ptr: u32,
}

impl RareSongHeader {
    /// Parse a 20-byte (`0x14`) little-endian Rareware song header.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < RARE_SONG_HEADER_SIZE {
            return Err(Error::UnexpectedEof);
        }
        let num_tracks = read_u32_le(data, 0x00);
        let ticks_per_quarter = read_u32_le(data, 0x04);
        let track_ptr_table = read_u32_le(data, 0x08);
        let instrument_table_ptr = read_u32_le(data, 0x0C);
        let percussion_table_ptr = read_u32_le(data, 0x10);

        if num_tracks == 0 || num_tracks > 64 {
            return Err(Error::InvalidHeader("num_tracks out of 1..=64"));
        }
        if ticks_per_quarter == 0 {
            return Err(Error::InvalidHeader("ticks_per_quarter == 0"));
        }
        if track_ptr_table < GBA_ROM_BUS_BASE
            || instrument_table_ptr < GBA_ROM_BUS_BASE
            || percussion_table_ptr < GBA_ROM_BUS_BASE
        {
            return Err(Error::InvalidHeader("song table pointer outside ROM bus"));
        }

        Ok(Self {
            num_tracks,
            ticks_per_quarter,
            track_ptr_table,
            instrument_table_ptr,
            percussion_table_ptr,
        })
    }
}

/// Zero-allocation borrowed view of a Rareware signed 8-bit PCM sample (`#![no_std]` friendly).
///
/// The `pcm` slice holds raw signed 8-bit PCM bytes (`i8` encoded as `u8`). If `pcm.len() > sample_len`,
/// byte `pcm[sample_len]` serves as the hardware 1-byte linear-interpolation guard byte read by
/// `ldrsb r2, [r0, #1]` when `pos == sample_len - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RareSampleView<'a> {
    /// Logical sample index.
    pub index: usize,
    /// Native sample rate in Hz.
    pub sample_rate_hz: u32,
    /// MIDI root note (`0..=127`).
    pub root_note: u8,
    /// Playback mode (`1..=4`).
    pub playback_mode: RarePlaybackMode,
    /// Loop length in bytes (`0` if one-shot).
    pub loop_length: u32,
    /// Logical sample length in bytes (excluding any trailing 1-byte interpolation guard).
    pub sample_len: usize,
    /// Raw signed 8-bit PCM bytes (length `>= sample_len`, ideally `sample_len + 1` with guard).
    pub pcm: &'a [u8],
}

impl<'a> RareSampleView<'a> {
    /// Wrap a raw signed 8-bit PCM byte slice directly without heap allocation.
    ///
    /// If `has_guard_byte` is `true` and `pcm.len() > 1`, `sample_len` is `pcm.len() - 1`
    /// and the final byte is used as the ARM `ldrsb r2, [r0, #1]` interpolation guard.
    #[must_use]
    pub const fn from_signed_pcm(
        index: usize,
        sample_rate_hz: u32,
        root_note: u8,
        playback_mode: RarePlaybackMode,
        loop_length: u32,
        pcm: &'a [u8],
        has_guard_byte: bool,
    ) -> Self {
        let sample_len = if has_guard_byte && !pcm.is_empty() {
            pcm.len() - 1
        } else {
            pcm.len()
        };
        Self {
            index,
            sample_rate_hz,
            root_note,
            playback_mode,
            loop_length,
            sample_len,
            pcm,
        }
    }

    /// Construct a [`RareSampleView`] from a parsed [`RareSampleDesc`] and the full GBA ROM slice.
    pub fn from_rom_descriptor(index: usize, desc: &RareSampleDesc, rom: &'a [u8]) -> Result<Self> {
        let start = desc.rom_pcm_offset();
        let len = desc.sample_len();
        let end = start.checked_add(len).ok_or(Error::UnexpectedEof)?;
        if end > rom.len() {
            return Err(Error::UnexpectedEof);
        }
        // Include the 1-byte interpolation guard if present in ROM
        let slice_end = if end < rom.len() { end + 1 } else { end };
        Ok(Self {
            index,
            sample_rate_hz: desc.sample_rate_hz,
            root_note: desc.root_note,
            playback_mode: desc.playback_mode,
            loop_length: desc.loop_length,
            sample_len: len,
            pcm: &rom[start..slice_end],
        })
    }

    /// Logical sample length in bytes (excluding any trailing interpolation guard).
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sample_len
    }

    /// Returns `true` if the sample has zero length.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sample_len == 0
    }

    /// Returns `true` if this sample loops (`MidiLooped` or `SfxLooped` with `loop_length > 0`).
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.playback_mode.is_looped() && self.loop_length > 0
    }

    /// Read signed 8-bit sample at `pos` (`i8`), returning `0` if out of bounds.
    #[inline]
    #[must_use]
    pub fn sample_i8(&self, pos: usize) -> i8 {
        self.pcm.get(pos).copied().map_or(0, |b| b as i8)
    }

    /// Read adjacent pair `(pcm[pos], pcm[pos + 1])` as `i32` for ARM 23-bit linear interpolation.
    ///
    /// Uses the explicit 1-byte guard at `pcm[sample_len]` when present; otherwise wraps to
    /// `loop_start` for looped samples or repeats `0` at end of one-shot samples.
    #[inline]
    #[must_use]
    pub fn sample_pair_i32(&self, pos: usize) -> (i32, i32) {
        let s0 = self.sample_i8(pos) as i32;
        let s1 = if pos + 1 < self.pcm.len() {
            self.pcm[pos + 1] as i8 as i32
        } else if self.is_looped() && self.loop_length > 0 {
            let loop_start = self.sample_len.saturating_sub(self.loop_length as usize);
            self.sample_i8(loop_start) as i32
        } else {
            0
        };
        (s0, s1)
    }
}

/// Heap-allocated Rareware GBA signed 8-bit PCM sample (`alloc` feature).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RareSample {
    /// Logical sample index.
    pub index: usize,
    /// Native sample rate in Hz.
    pub sample_rate_hz: u32,
    /// MIDI root note (`0..=127`).
    pub root_note: u8,
    /// Playback mode (`1..=4`).
    pub playback_mode: RarePlaybackMode,
    /// Loop length in bytes (`0` if one-shot).
    pub loop_length: u32,
    /// Logical sample length in bytes.
    pub sample_len: usize,
    /// Owned signed 8-bit PCM bytes (including optional trailing guard byte).
    pub pcm: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl RareSample {
    /// Borrow this owned sample as a zero-allocation [`RareSampleView`].
    #[must_use]
    pub fn as_view(&self) -> RareSampleView<'_> {
        RareSampleView {
            index: self.index,
            sample_rate_hz: self.sample_rate_hz,
            root_note: self.root_note,
            playback_mode: self.playback_mode,
            loop_length: self.loop_length,
            sample_len: self.sample_len,
            pcm: &self.pcm,
        }
    }

    /// Construct an owned [`RareSample`] from a [`RareSampleView`].
    #[must_use]
    pub fn from_view(view: RareSampleView<'_>) -> Self {
        Self {
            index: view.index,
            sample_rate_hz: view.sample_rate_hz,
            root_note: view.root_note,
            playback_mode: view.playback_mode,
            loop_length: view.loop_length,
            sample_len: view.sample_len,
            pcm: view.pcm.to_vec(),
        }
    }
}

#[inline]
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}
