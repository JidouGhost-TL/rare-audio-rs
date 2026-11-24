use std::hint::black_box;
use std::time::Instant;

use rare_audio::{
    AccuracySynth, RareAudioVersion, RarePlaybackMode, RareSampleView, RareVoice, StudioSynth,
    RARE_DEFAULT_FRAME_SAMPLES, RARE_DEFAULT_MIX_RATE,
};

fn main() {
    println!("=== rare-audio Performance Benchmarks ===");

    let pcm_i8: Vec<u8> = (0i32..4097)
        .map(|i| (((i % 32) - 16) * 7).clamp(-128, 127) as i8 as u8)
        .collect();
    let sample = RareSampleView::from_signed_pcm(
        0,
        RARE_DEFAULT_MIX_RATE,
        60,
        RarePlaybackMode::MidiLooped,
        2048,
        &pcm_i8,
        true,
    );

    let mut acc = vec![0i16; RARE_DEFAULT_FRAME_SAMPLES];
    let mut dma = vec![0i8; RARE_DEFAULT_FRAME_SAMPLES];
    let mut out_l = vec![0.0f32; 1024];
    let mut out_r = vec![0.0f32; 1024];

    let iters = 200_000usize;

    // 1. AccuracySynth (StandardSingleClamp, 1:1, 224 samples)
    for _ in 0..1000 {
        let mut v = RareVoice::with_phase(0, RarePlaybackMode::MidiLooped, 0, 0, 0x0080_0000, 128);
        AccuracySynth::mix_voice_view_version(
            &sample,
            &mut v,
            &mut acc,
            true,
            RareAudioVersion::StandardSingleClamp,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut v = RareVoice::with_phase(0, RarePlaybackMode::MidiLooped, 0, 0, 0x0080_0000, 128);
        AccuracySynth::mix_voice_view_version(
            black_box(&sample),
            &mut v,
            &mut acc,
            true,
            RareAudioVersion::StandardSingleClamp,
        );
        black_box(&acc);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth Standard (1:1, 224f):        {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * RARE_DEFAULT_FRAME_SAMPLES) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 2. AccuracySynth (StandardSingleClamp, resampled Q9.23, 224 samples)
    let start = Instant::now();
    for _ in 0..iters {
        let mut v = RareVoice::with_phase(
            0,
            RarePlaybackMode::MidiLooped,
            3,
            0x0023_4567,
            0x00B5_04F3,
            128,
        );
        AccuracySynth::mix_voice_view_version(
            black_box(&sample),
            &mut v,
            &mut acc,
            true,
            RareAudioVersion::StandardSingleClamp,
        );
        black_box(&acc);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth Standard (resampled, 224f):  {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * RARE_DEFAULT_FRAME_SAMPLES) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 3. AccuracySynth (PackedFastPath / Banjo-Pilot, resampled, 224 samples)
    let start = Instant::now();
    for _ in 0..iters {
        let mut v = RareVoice::with_phase(
            0,
            RarePlaybackMode::MidiLooped,
            3,
            0x0023_4567,
            0x00B5_04F3,
            128,
        );
        AccuracySynth::mix_voice_view_version(
            black_box(&sample),
            &mut v,
            &mut acc,
            true,
            RareAudioVersion::PackedFastPath,
        );
        black_box(&acc);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth PackedFastPath (224f):       {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * RARE_DEFAULT_FRAME_SAMPLES) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 4. AccuracySynth::clamp_i16_to_dma8 (224 samples)
    let start = Instant::now();
    for _ in 0..iters {
        AccuracySynth::clamp_i16_to_dma8(black_box(&acc), &mut dma);
        black_box(&dma);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth clamp_i16_to_dma8 (224f):    {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * RARE_DEFAULT_FRAME_SAMPLES) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 5. StudioSynth (48 kHz Lanczos-3 stereo, 1024 frames)
    let studio = StudioSynth::new(48_000);
    let studio_iters = 20_000usize;
    let start = Instant::now();
    for _ in 0..studio_iters {
        out_l.fill(0.0);
        out_r.fill(0.0);
        let mut v = RareVoice::with_phase(
            0,
            RarePlaybackMode::MidiLooped,
            0,
            0x0012_3456,
            0x00B5_04F3,
            128,
        );
        v.pan = 0.25;
        studio.mix_voice_view_stereo(black_box(&sample), &mut v, &mut out_l, &mut out_r);
        black_box((&out_l, &out_r));
    }
    let elapsed = start.elapsed();
    println!(
        "StudioSynth Lanczos-3 Stereo (48k, 1024f): {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / studio_iters as f64,
        ((studio_iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );
}
