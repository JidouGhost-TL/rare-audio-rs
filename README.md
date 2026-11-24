# rare-audio-rs

**Zero-dependency, pure Rust parser, software synthesizer, and byte-exact golden-tested emulator for the Rareware GBA Sound Engine (Game Boy Advance).**

[![Crates.io](https://img.shields.io/crates/v/rare-audio.svg)](https://crates.io/crates/rare-audio)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#features)

The **Rareware GBA Sound Engine** is a custom ARM7TDMI IWRAM-accelerated DirectSound A software mixer powering Rare's entire Game Boy Advance catalog — including *Donkey Kong Country 1–3*, *Banjo-Kazooie: Grunty's Revenge*, *Banjo-Pilot*, *Sabre Wulf*, *It's Mr. Pants*, and unreleased prototypes (*Battletoads*, *Diddy Kong Pilot*). Running at **13,379 Hz** (`REG_TM0CNT_L = 0xFB1A`, 224 mono samples/frame via `DMA1`), it mixes up to 8 voices of raw signed 8-bit PCM (`i8`) using **23-bit fractional linear interpolation** (`Q9.23` phase accumulator).

`rare-audio-rs` provides a clean-room, reverse-engineered implementation designed for audio preservation, DAWs, PC source ports, and emulators.

> **Looking for a ready-to-use CLI?** You can scan GBA ROMs, extract WAV samples, and compile SoundFont 2.04 (.sf2) banks directly from the command line using our **[gbakit](https://github.com/JidouGhost-TL/gbakit)** toolkit (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

---

## Features

- **Zero External Dependencies**: Pure Rust with no dependencies in `[dependencies]`, `#![deny(unsafe_code)]`.
- **`#![no_std]` Compatible**: Ready for embedded targets and WASM runtimes (`alloc` optional).
- **Native Rareware Binary Parsers**: Zero-allocation `RareSampleDesc` (68-byte / `0x44` descriptor), `RareSongHeader` (20-byte / `0x14` song header), and `RareSampleView` (signed 8-bit PCM with 1-byte linear-interpolation guard).
- **Dual Synthesis Engines**:
  - **`AccuracySynth`**: 100% bit-exact emulation of all 4 Rareware GBA ARM7TDMI IWRAM mixer revisions — exact Q9.23 `asr #23` linear interpolation, `(interp * vol) >> 7` scaling, loop wrap (`pos -= loop_length`), one-shot early exit, *Banjo-Pilot*'s lookahead-guarded 2x-packed fast path (including its authentic ARM `add r2, sl, r2, lsl #16` negative-low-halfword `-1` bleed), and DirectSound A `i16 -> i8` clamping.
  - **`StudioSynth`**: Renders at 48,000 Hz, 32-bit floating-point stereo with a compile-time normalized 1,024-phase 6-tap Lanczos-3 windowed-sinc polyphase filter and true equal-power stereo panning.
- **Byte-Exact mGBA ARM Golden Tested**: **2,688 / 2,688** output bytes verified strictly byte-for-byte (`assert_eq!`) across all 4 distinct Rareware ARM mixer revisions discovered across all 26 Rareware GBA ROMs.

---

## Supported Rareware GBA Mixer Revisions

Auditing all 26 unique Rareware GBA ROMs across the full catalog identified exactly **4 distinct ARM7TDMI IWRAM mixer revisions**, all supported and byte-exact golden-tested via `RareAudioVersion` (with automatic detection via `RareAudioVersion::detect_from_bytes`):

| Revision | Corpus Share & Representative Titles | Inner Mix Loop | Clamp Kernel | Bit-Exact Quirk |
| --- | --- | --- | --- | --- |
| `StandardSingleClamp` | 15 ROMs (*Donkey Kong Country* US/EU/JP, *Banjo-Kazooie: Grunty's Revenge* + betas, *Sabre Wulf*, *It's Mr. Pants*, *Battletoads* proto, *Diddy Kong Pilot* proto) | Per-sample `strh` / `ldrsh` (`Q9.23`) | `SingleByteLdrsh` (`ldrsh` -> `strb`) | Exact clean 16-bit write/accumulate |
| `StandardDualHalfClamp` | 3 ROMs (*Donkey Kong Country 2* US/EU/JP) | Per-sample `strh` / `ldrsh` (`Q9.23`) | `DualHalfCmn` (`mov r4, #0x800000` -> `strh`) | 2x-packed halfword clamp |
| `StandardDualByteClamp` | 4 ROMs (*Donkey Kong Country 3* US/EU/JP + proto) | Per-sample `strh` / `ldrsh` (`Q9.23`) | `DualByteLdr32` (`ldr` -> `strb` x2) | 2x-unrolled byte clamp |
| `PackedFastPath` | 4 ROMs (*Banjo-Pilot* US/EU + 2 protos) | Lookahead 2x-packed `str` (first voice) + 4x `ror #16` accumulate + `strh` slow fallback | `DualHalfCmn` (`mov r4, #0x800000` -> `strh`) | First-voice fast path `add r2, sl, r2, lsl #16` subtracts `1` from odd sample `2k+1` whenever even sample `2k < 0` |

---

## Quickstart

### Zero-Allocation Mixing (`#![no_std]`)

```rust
use rare_audio::{
    AccuracySynth, RareAudioVersion, RarePlaybackMode, RareSampleView, RareVoice,
    RARE_DEFAULT_FRAME_SAMPLES, RARE_DEFAULT_MIX_RATE,
};

// Signed 8-bit PCM with 1-byte interpolation guard
let pcm_i8: &[u8] = &[0, 32, 64, 96, 64, 32, 0, 224, 192, 160, 192, 224, 0];
let sample = RareSampleView::from_signed_pcm(
    0,
    RARE_DEFAULT_MIX_RATE,
    60,
    RarePlaybackMode::MidiLooped,
    12,
    pcm_i8,
    true,
);

let mut voice = RareVoice::new(0, RarePlaybackMode::MidiLooped, RARE_DEFAULT_MIX_RATE, 128);
let mut acc = [0i16; RARE_DEFAULT_FRAME_SAMPLES];
let mut dma = [0i8; RARE_DEFAULT_FRAME_SAMPLES];

AccuracySynth::mix_voice_view_version(
    &sample,
    &mut voice,
    &mut acc,
    true,
    RareAudioVersion::StandardSingleClamp,
);
AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
```

---

## Golden Test Methodology

Every mixing and clamping path in `AccuracySynth` is verified strictly byte-for-byte against real ARM7TDMI execution captured in headless `libmgba.so.0.10` using clean-room synthetic deterministic waveforms:

1. **Tier 1 (Binary Descriptors)**: 68-byte `RareSampleDesc` and 20-byte `RareSongHeader` parsing, ADSR/pointer validation, and loop-bound calculation.
2. **Tier 2 (Timer & Fixed-Point Math)**: GBA Timer 0 reload (`0xFB1A` -> 13,379 Hz), Q9.23 phase step, Q16.16 pitch scale, and ARM volume shift chain.
3. **Tier 3 (2,688 / 2,688 Byte-Exact ARM Mixer Goldens)**: Strict `assert_eq!` across 12 controlled ARM execution vectors covering all 4 Rareware GBA mixer revisions (`StandardSingleClamp`, `StandardDualHalfClamp`, `StandardDualByteClamp`, and `PackedFastPath` — including the 58-sample negative-low-halfword `-1` bleed verification and lookahead slow-path fallback).

---

## Benchmarks

```sh
cargo bench --bench bench_audio
```

| Benchmark | Throughput |
| --- | ---: |
| `AccuracySynth::mix_voice_view` (Standard 1:1, 224f) | ~19.31 GSamples/s |
| `AccuracySynth::mix_voice_view` (Standard resampled, 224f) | ~1.17 GSamples/s |
| `AccuracySynth::mix_voice_view_version` (`PackedFastPath`, 224f) | ~1.13 GSamples/s |
| `AccuracySynth::clamp_i16_to_dma8` (224f) | ~46.28 GSamples/s |
| `StudioSynth::mix_voice_view_stereo` (48 kHz Lanczos-3, 1,024f) | ~400.5 MSamples/s |

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
