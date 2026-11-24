//! Byte-exact mGBA ARM Golden Integration Tests for the Rareware GBA Sound Engine (`rare-audio`).
//!
//! Verifies:
//! - **Tier 1**: Native 68-byte (`0x44`) `RareSampleDesc` and 20-byte (`0x14`) `RareSongHeader`
//!   binary parsing, loop-bound validation (`loop_start = sample_len - loop_length`), and 1-byte
//!   interpolation guard handling.
//! - **Tier 2**: GBA Timer 0 hardware rate (`0xFB1A` -> 13,379 Hz), Q9.23 phase stepping,
//!   Q16.16 pitch scaling, ARM volume shift chain, and `RareAudioVersion::detect_from_bytes`.
//! - **Tier 3**: Strict byte-for-byte `assert_eq!` (`0` tolerance, 2,688 / 2,688 bytes) of
//!   `AccuracySynth` against real ARM7TDMI execution captured in headless `libmgba.so.0.10`
//!   across all 4 Rareware GBA mixer revisions:
//!   1. `StandardSingleClamp` (*Donkey Kong Country 1* / *Banjo-Kazooie: GR* / *Sabre Wulf* /
//!      *It's Mr. Pants* / *Battletoads* / *Diddy Kong Pilot*): 1:1 linear, resampled Q9.23,
//!      loop wrap, one-shot mid-buffer end, 4-voice saturated clamp (5 vectors, 1,120 bytes).
//!   2. `StandardDualHalfClamp` (*Donkey Kong Country 2*): resampled + 3-voice saturated
//!      `DualHalfCmn` clamp (2 vectors, 448 bytes).
//!   3. `StandardDualByteClamp` (*Donkey Kong Country 3*): resampled + 3-voice saturated
//!      `DualByteLdr32` clamp (2 vectors, 448 bytes).
//!   4. `PackedFastPath` (*Banjo-Pilot*): 2x-packed first-voice fast path (verifying authentic
//!      ARM negative-low-halfword `-1` bleed on 58 odd samples), 3-voice fast path + 4x
//!      accumulate, and lookahead slow-path fallback on loop wrap (3 vectors, 672 bytes).

use rare_audio::{
    apply_pitch_scale_q16, compute_effective_volume, freq_to_step_q23, note_to_step_q23,
    rate_from_timer_reload, AccuracySynth, RareAudioVersion, RareClampMode, RarePlaybackMode,
    RareSampleDesc, RareSampleView, RareSongHeader, RareVoice, StudioSynth, GBA_ROM_BUS_BASE,
    RARE_DEFAULT_FRAME_SAMPLES, RARE_DEFAULT_MIX_RATE, RARE_DESC_MAGIC_SIZE, RARE_FRAC_ONE,
    RARE_TIMER0_RELOAD,
};

const MANIFEST_JSON: &str = include_str!("golden/manifest.json");

const GOLDEN_DKC1_1TO1: &[u8] = include_bytes!("golden/dkc1_linear_1to1.bin");
const GOLDEN_DKC1_RESAMPLED: &[u8] = include_bytes!("golden/dkc1_linear_resampled.bin");
const GOLDEN_DKC1_LOOP: &[u8] = include_bytes!("golden/dkc1_loop_wrap.bin");
const GOLDEN_DKC1_ONESHOT: &[u8] = include_bytes!("golden/dkc1_oneshot_end.bin");
const GOLDEN_DKC1_MULTI: &[u8] = include_bytes!("golden/dkc1_multi_clamped.bin");

const GOLDEN_DKC2_RESAMPLED: &[u8] = include_bytes!("golden/dkc2_resampled_dualhalf.bin");
const GOLDEN_DKC2_MULTI: &[u8] = include_bytes!("golden/dkc2_multi_clamped_dualhalf.bin");

const GOLDEN_DKC3_RESAMPLED: &[u8] = include_bytes!("golden/dkc3_resampled_dualbyte.bin");
const GOLDEN_DKC3_MULTI: &[u8] = include_bytes!("golden/dkc3_multi_clamped_dualbyte.bin");

const GOLDEN_BAJE_BLEED: &[u8] = include_bytes!("golden/baje_fastpath_bleed.bin");
const GOLDEN_BAJE_MULTI: &[u8] = include_bytes!("golden/baje_fastpath_multi.bin");
const GOLDEN_BAJE_SLOW_LOOP: &[u8] = include_bytes!("golden/baje_slowpath_loop.bin");

/// Clean-room synthetic signed 8-bit triangle wave (length + 1 guard byte).
fn fill_triangle<const N: usize>(buf: &mut [u8; N]) {
    for (i, slot) in buf.iter_mut().enumerate() {
        let p = (i % 32) as i32;
        let v = if p < 8 {
            p * 15
        } else if p < 24 {
            (16 - p) * 15
        } else {
            (p - 32) * 15
        };
        *slot = v.clamp(-128, 127) as i8 as u8;
    }
}

/// Clean-room synthetic signed 8-bit sawtooth wave (length + 1 guard byte).
fn fill_saw<const N: usize>(buf: &mut [u8; N]) {
    for (i, slot) in buf.iter_mut().enumerate() {
        let v = (((i % 25) as i32) - 12) * 9;
        *slot = v.clamp(-128, 127) as i8 as u8;
    }
}

/// Clean-room synthetic signed 8-bit negative-heavy asymmetric wave (length + 1 guard byte).
fn fill_neg_heavy<const N: usize>(buf: &mut [u8; N]) {
    for (i, slot) in buf.iter_mut().enumerate() {
        let v = -90 + (((i * 19) % 170) as i32);
        *slot = v.clamp(-128, 127) as i8 as u8;
    }
}

fn assert_mono_golden_exact(dma_out: &[i8], expected: &[u8], label: &str) {
    assert_eq!(dma_out.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &exp)) in dma_out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, exp as i8,
            "{label}: byte mismatch at sample {i} (got {got}, expected {})",
            exp as i8
        );
    }
}

#[test]
fn test_manifest_integrity() {
    assert!(MANIFEST_JSON.contains("\"total_bytes\": 2688"));
    for name in [
        "dkc1_linear_1to1.bin",
        "dkc1_linear_resampled.bin",
        "dkc1_loop_wrap.bin",
        "dkc1_oneshot_end.bin",
        "dkc1_multi_clamped.bin",
        "dkc2_resampled_dualhalf.bin",
        "dkc2_multi_clamped_dualhalf.bin",
        "dkc3_resampled_dualbyte.bin",
        "dkc3_multi_clamped_dualbyte.bin",
        "baje_fastpath_bleed.bin",
        "baje_fastpath_multi.bin",
        "baje_slowpath_loop.bin",
    ] {
        assert!(MANIFEST_JSON.contains(name), "manifest missing {name}");
    }
}

// ============================================================================
// Tier 1: Native 68-byte RareSampleDesc & 20-byte RareSongHeader Parsing
// ============================================================================

#[test]
fn test_tier1_sample_desc_and_song_header_parse() {
    let mut raw_desc = [0u8; 68];
    let words: [u32; 17] = [
        RARE_DESC_MAGIC_SIZE,
        2,                        // MidiLooped
        13_379,                   // sample_rate_hz
        60,                       // root_note (C4)
        GBA_ROM_BUS_BASE + 0x100, // sample_start_ptr
        48,                       // loop_length
        GBA_ROM_BUS_BASE + 0x180, // sample_end_ptr (len = 128)
        0,                        // reserved_0
        0,                        // reserved_1
        99,                       // attack
        20,                       // decay
        85,                       // sustain
        35,                       // release
        (-12i32) as u32,          // fine_tune_cents
        2,                        // pitch_bend_range
        5,                        // vibrato_rate
        10,                       // vibrato_depth
    ];
    for (i, &w) in words.iter().enumerate() {
        raw_desc[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
    }

    let desc = RareSampleDesc::parse(&raw_desc).expect("valid descriptor");
    assert_eq!(desc.playback_mode, RarePlaybackMode::MidiLooped);
    assert_eq!(desc.sample_rate_hz, 13_379);
    assert_eq!(desc.root_note, 60);
    assert_eq!(desc.sample_len(), 128);
    assert!(desc.is_looped());
    assert_eq!(desc.loop_start(), Some(80));
    assert_eq!(desc.fine_tune_cents, -12);
    assert_eq!(desc.adsr.attack, 99);
    assert_eq!(desc.adsr.sustain, 85);

    // Song header parse
    let mut raw_song = [0u8; 20];
    let song_words: [u32; 5] = [
        8,
        384,
        GBA_ROM_BUS_BASE + 0x1000,
        GBA_ROM_BUS_BASE + 0x2000,
        GBA_ROM_BUS_BASE + 0x3000,
    ];
    for (i, &w) in song_words.iter().enumerate() {
        raw_song[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
    }
    let song = RareSongHeader::parse(&raw_song).expect("valid song header");
    assert_eq!(song.num_tracks, 8);
    assert_eq!(song.ticks_per_quarter, 384);
}

// ============================================================================
// Tier 2: Hardware Timer, Q9.23 Phase Math & Volume Shift Chain
// ============================================================================

#[test]
fn test_tier2_timer_and_fixed_point_math() {
    assert_eq!(
        rate_from_timer_reload(RARE_TIMER0_RELOAD),
        RARE_DEFAULT_MIX_RATE
    );
    assert_eq!(freq_to_step_q23(RARE_DEFAULT_MIX_RATE), RARE_FRAC_ONE);
    assert_eq!(
        note_to_step_q23(13_379, 60, 72, 13_379),
        RARE_FRAC_ONE * 2
    );
    assert_eq!(
        apply_pitch_scale_q16(RARE_FRAC_ONE, 0x0001_8000),
        (RARE_FRAC_ONE * 3) / 2
    );
    assert_eq!(compute_effective_volume(128, Some(127), 128, 1 << 19), 128);
    assert_eq!(
        RareAudioVersion::PackedFastPath.clamp_mode(),
        RareClampMode::DualHalfCmn
    );
}

// ============================================================================
// Tier 3: 2,688 / 2,688 Byte-Exact libmgba ARM7TDMI Golden Tests (All 4 Revisions)
// ============================================================================

#[test]
fn test_tier3_dkc1_standard_single_clamp_all_5_vectors() {
    let mut tri512 = [0u8; 513];
    let mut saw512 = [0u8; 513];
    let mut neg512 = [0u8; 513];
    let mut tri96 = [0u8; 97];
    fill_triangle(&mut tri512);
    fill_saw(&mut saw512);
    fill_neg_heavy(&mut neg512);
    fill_triangle(&mut tri96);

    let v_tri512 = RareSampleView::from_signed_pcm(
        0,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &tri512,
        true,
    );
    let v_saw512 = RareSampleView::from_signed_pcm(
        1,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &saw512,
        true,
    );
    let v_neg512 = RareSampleView::from_signed_pcm(
        2,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &neg512,
        true,
    );
    let v_tri96_loop = RareSampleView::from_signed_pcm(
        3,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiLooped,
        64,
        &tri96,
        true,
    );
    let v_tri96_oneshot = RareSampleView::from_signed_pcm(
        4,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &tri96,
        true,
    );

    let ver = RareAudioVersion::StandardSingleClamp;
    let mut acc = [0i16; RARE_DEFAULT_FRAME_SAMPLES];
    let mut dma = [0i8; RARE_DEFAULT_FRAME_SAMPLES];

    // Vector 1: dkc1_linear_1to1
    let mut v1 = RareVoice::with_phase(0, RarePlaybackMode::MidiOneShot, 0, 0, 0x0080_0000, 128);
    AccuracySynth::mix_voice_view_version(&v_tri512, &mut v1, &mut acc, true, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC1_1TO1, "dkc1_linear_1to1");

    // Vector 2: dkc1_linear_resampled
    let mut v2 = RareVoice::with_phase(
        0,
        RarePlaybackMode::MidiOneShot,
        3,
        0x0023_4567,
        0x00B5_04F3,
        128,
    );
    AccuracySynth::mix_voice_view_version(&v_tri512, &mut v2, &mut acc, true, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC1_RESAMPLED, "dkc1_linear_resampled");

    // Vector 3: dkc1_loop_wrap
    let mut v3 = RareVoice::with_phase(
        3,
        RarePlaybackMode::MidiLooped,
        40,
        0x0040_0000,
        0x00A1_2345,
        112,
    );
    AccuracySynth::mix_voice_view_version(&v_tri96_loop, &mut v3, &mut acc, true, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC1_LOOP, "dkc1_loop_wrap");
    assert!(v3.active);

    // Vector 4: dkc1_oneshot_end
    let mut v4a = RareVoice::with_phase(1, RarePlaybackMode::MidiOneShot, 0, 0, 0x0080_0000, 64);
    let mut v4b = RareVoice::with_phase(
        4,
        RarePlaybackMode::MidiOneShot,
        20,
        0x0010_0000,
        0x00C0_0000,
        96,
    );
    AccuracySynth::mix_voice_view_version(&v_saw512, &mut v4a, &mut acc, true, ver);
    AccuracySynth::mix_voice_view_version(&v_tri96_oneshot, &mut v4b, &mut acc, false, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC1_ONESHOT, "dkc1_oneshot_end");
    assert!(!v4b.active, "one-shot voice should deactivate at sample_end");

    // Vector 5: dkc1_multi_clamped
    let mut v5a = RareVoice::with_phase(
        0,
        RarePlaybackMode::MidiOneShot,
        0,
        0x0012_3456,
        0x0090_0000,
        128,
    );
    let mut v5b = RareVoice::with_phase(
        1,
        RarePlaybackMode::MidiOneShot,
        11,
        0x0065_4321,
        0x00B0_0000,
        128,
    );
    let mut v5c = RareVoice::with_phase(
        2,
        RarePlaybackMode::MidiOneShot,
        5,
        0x0033_3333,
        0x00A0_0000,
        128,
    );
    let mut v5d = RareVoice::with_phase(3, RarePlaybackMode::MidiLooped, 0, 0, 0x00D0_0000, 128);
    AccuracySynth::mix_voice_view_version(&v_tri512, &mut v5a, &mut acc, true, ver);
    AccuracySynth::mix_voice_view_version(&v_saw512, &mut v5b, &mut acc, false, ver);
    AccuracySynth::mix_voice_view_version(&v_neg512, &mut v5c, &mut acc, false, ver);
    AccuracySynth::mix_voice_view_version(&v_tri96_loop, &mut v5d, &mut acc, false, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC1_MULTI, "dkc1_multi_clamped");
}

#[test]
fn test_tier3_dkc2_standard_dual_half_clamp() {
    let mut tri512 = [0u8; 513];
    let mut saw512 = [0u8; 513];
    let mut neg512 = [0u8; 513];
    fill_triangle(&mut tri512);
    fill_saw(&mut saw512);
    fill_neg_heavy(&mut neg512);

    let v_tri = RareSampleView::from_signed_pcm(
        0,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &tri512,
        true,
    );
    let v_saw = RareSampleView::from_signed_pcm(
        1,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &saw512,
        true,
    );
    let v_neg = RareSampleView::from_signed_pcm(
        2,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &neg512,
        true,
    );

    let ver = RareAudioVersion::StandardDualHalfClamp;
    let mut acc = [0i16; RARE_DEFAULT_FRAME_SAMPLES];
    let mut dma = [0i8; RARE_DEFAULT_FRAME_SAMPLES];

    // Vector 6: dkc2_resampled_dualhalf
    let mut v6 = RareVoice::with_phase(
        2,
        RarePlaybackMode::MidiOneShot,
        7,
        0x0011_2233,
        0x0098_7654,
        120,
    );
    AccuracySynth::mix_voice_view_version(&v_neg, &mut v6, &mut acc, true, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC2_RESAMPLED, "dkc2_resampled_dualhalf");

    // Vector 7: dkc2_multi_clamped_dualhalf
    let mut v7a = RareVoice::with_phase(0, RarePlaybackMode::MidiOneShot, 0, 0, 0x0080_0000, 128);
    let mut v7b = RareVoice::with_phase(
        1,
        RarePlaybackMode::MidiOneShot,
        4,
        0x0044_4444,
        0x00A8_0000,
        128,
    );
    let mut v7c = RareVoice::with_phase(
        2,
        RarePlaybackMode::MidiOneShot,
        2,
        0x0022_2222,
        0x0094_0000,
        128,
    );
    AccuracySynth::mix_voice_view_version(&v_tri, &mut v7a, &mut acc, true, ver);
    AccuracySynth::mix_voice_view_version(&v_saw, &mut v7b, &mut acc, false, ver);
    AccuracySynth::mix_voice_view_version(&v_neg, &mut v7c, &mut acc, false, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC2_MULTI, "dkc2_multi_clamped_dualhalf");
}

#[test]
fn test_tier3_dkc3_standard_dual_byte_clamp() {
    let mut tri512 = [0u8; 513];
    let mut saw512 = [0u8; 513];
    let mut neg512 = [0u8; 513];
    fill_triangle(&mut tri512);
    fill_saw(&mut saw512);
    fill_neg_heavy(&mut neg512);

    let v_tri = RareSampleView::from_signed_pcm(
        0,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &tri512,
        true,
    );
    let v_saw = RareSampleView::from_signed_pcm(
        1,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &saw512,
        true,
    );
    let v_neg = RareSampleView::from_signed_pcm(
        2,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &neg512,
        true,
    );

    let ver = RareAudioVersion::StandardDualByteClamp;
    let mut acc = [0i16; RARE_DEFAULT_FRAME_SAMPLES];
    let mut dma = [0i8; RARE_DEFAULT_FRAME_SAMPLES];

    // Vector 8: dkc3_resampled_dualbyte
    let mut v8 = RareVoice::with_phase(
        1,
        RarePlaybackMode::MidiOneShot,
        13,
        0x0055_6677,
        0x00C4_0000,
        104,
    );
    AccuracySynth::mix_voice_view_version(&v_saw, &mut v8, &mut acc, true, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC3_RESAMPLED, "dkc3_resampled_dualbyte");

    // Vector 9: dkc3_multi_clamped_dualbyte
    let mut v9a = RareVoice::with_phase(
        0,
        RarePlaybackMode::MidiOneShot,
        2,
        0x0010_1010,
        0x0088_0000,
        128,
    );
    let mut v9b = RareVoice::with_phase(
        2,
        RarePlaybackMode::MidiOneShot,
        0,
        0x0070_0000,
        0x009C_0000,
        128,
    );
    let mut v9c = RareVoice::with_phase(
        1,
        RarePlaybackMode::MidiOneShot,
        9,
        0x0030_3030,
        0x00B4_0000,
        128,
    );
    AccuracySynth::mix_voice_view_version(&v_tri, &mut v9a, &mut acc, true, ver);
    AccuracySynth::mix_voice_view_version(&v_neg, &mut v9b, &mut acc, false, ver);
    AccuracySynth::mix_voice_view_version(&v_saw, &mut v9c, &mut acc, false, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_DKC3_MULTI, "dkc3_multi_clamped_dualbyte");
}

#[test]
fn test_tier3_banjo_pilot_packed_fast_path_bleed_and_fallback() {
    let mut tri512 = [0u8; 513];
    let mut saw512 = [0u8; 513];
    let mut neg512 = [0u8; 513];
    let mut tri96 = [0u8; 97];
    fill_triangle(&mut tri512);
    fill_saw(&mut saw512);
    fill_neg_heavy(&mut neg512);
    fill_triangle(&mut tri96);

    let v_tri = RareSampleView::from_signed_pcm(
        0,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &tri512,
        true,
    );
    let v_saw = RareSampleView::from_signed_pcm(
        1,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &saw512,
        true,
    );
    let v_neg = RareSampleView::from_signed_pcm(
        2,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiOneShot,
        0,
        &neg512,
        true,
    );
    let v_tri96_loop = RareSampleView::from_signed_pcm(
        3,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiLooped,
        64,
        &tri96,
        true,
    );

    let ver = RareAudioVersion::PackedFastPath;
    let mut acc = [0i16; RARE_DEFAULT_FRAME_SAMPLES];
    let mut dma = [0i8; RARE_DEFAULT_FRAME_SAMPLES];

    // Vector 10: baje_fastpath_bleed (authentic ARM `add r2, sl, r2, lsl #16` -1 bleed on odd samples)
    let mut v10 = RareVoice::with_phase(
        2,
        RarePlaybackMode::MidiOneShot,
        0,
        0x0012_3456,
        0x0090_0000,
        128,
    );
    AccuracySynth::mix_voice_view_version(&v_neg, &mut v10, &mut acc, true, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_BAJE_BLEED, "baje_fastpath_bleed");

    // Verify PackedFastPath genuinely differs from StandardSingleClamp on 58 odd samples
    let mut v10_std = RareVoice::with_phase(
        2,
        RarePlaybackMode::MidiOneShot,
        0,
        0x0012_3456,
        0x0090_0000,
        128,
    );
    let mut acc_std = [0i16; RARE_DEFAULT_FRAME_SAMPLES];
    let mut dma_std = [0i8; RARE_DEFAULT_FRAME_SAMPLES];
    AccuracySynth::mix_voice_view_version(
        &v_neg,
        &mut v10_std,
        &mut acc_std,
        true,
        RareAudioVersion::StandardSingleClamp,
    );
    AccuracySynth::clamp_i16_to_dma8(&acc_std, &mut dma_std);
    let diffs = dma
        .iter()
        .zip(dma_std.iter())
        .filter(|(&a, &b)| a != b)
        .count();
    assert_eq!(diffs, 58, "expected exactly 58 ARM -1 bleed samples");

    // Vector 11: baje_fastpath_multi (Voice 0 fast-path overwrite + Voices 1 & 2 accumulate)
    let mut v11a = RareVoice::with_phase(
        2,
        RarePlaybackMode::MidiOneShot,
        0,
        0x0012_3456,
        0x0090_0000,
        110,
    );
    let mut v11b = RareVoice::with_phase(
        0,
        RarePlaybackMode::MidiOneShot,
        4,
        0x0040_0000,
        0x00A0_0000,
        100,
    );
    let mut v11c = RareVoice::with_phase(
        1,
        RarePlaybackMode::MidiOneShot,
        8,
        0x0020_0000,
        0x0085_0000,
        90,
    );
    AccuracySynth::mix_voice_view_version(&v_neg, &mut v11a, &mut acc, true, ver);
    AccuracySynth::mix_voice_view_version(&v_tri, &mut v11b, &mut acc, false, ver);
    AccuracySynth::mix_voice_view_version(&v_saw, &mut v11c, &mut acc, false, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_BAJE_MULTI, "baje_fastpath_multi");

    // Vector 12: baje_slowpath_loop (lookahead detects end crossing -> exact strh slow path)
    let mut v12 = RareVoice::with_phase(
        3,
        RarePlaybackMode::MidiLooped,
        40,
        0x0040_0000,
        0x00A1_2345,
        112,
    );
    AccuracySynth::mix_voice_view_version(&v_tri96_loop, &mut v12, &mut acc, true, ver);
    AccuracySynth::clamp_i16_to_dma8(&acc, &mut dma);
    assert_mono_golden_exact(&dma, GOLDEN_BAJE_SLOW_LOOP, "baje_slowpath_loop");
}

// ============================================================================
// StudioSynth 48 kHz Lanczos-3 & Equal-Power Panning Verification
// ============================================================================

#[test]
fn test_studio_synth_lanczos3_and_equal_power_panning() {
    let mut tri = [0u8; 513];
    fill_triangle(&mut tri);
    let sample = RareSampleView::from_signed_pcm(
        0,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiLooped,
        256,
        &tri,
        true,
    );

    let studio = StudioSynth::new(48_000);
    let mut voice = RareVoice::new(0, RarePlaybackMode::MidiLooped, RARE_DEFAULT_MIX_RATE, 128);
    voice.pan = -0.5;

    let mut out_l = [0.0f32; 256];
    let mut out_r = [0.0f32; 256];
    studio.mix_voice_view_stereo(&sample, &mut voice, &mut out_l, &mut out_r);

    let energy_l: f32 = out_l.iter().map(|x| x * x).sum();
    let energy_r: f32 = out_r.iter().map(|x| x * x).sum();
    assert!(energy_l > 0.1 && energy_r > 0.05);
    // For pan = -0.5: gain_l^2 = 0.75, gain_r^2 = 0.25 -> ratio = 3.0
    let ratio = energy_l / energy_r;
    assert!((ratio - 3.0).abs() < 1e-4, "expected 3.0 energy ratio, got {ratio}");
}
