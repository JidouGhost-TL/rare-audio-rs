//! Rareware GBA Sound Engine revision detection and ARM mixer variant metadata.
//!
//! Auditing all 26 unique Rareware GBA ROMs across the full GBA catalog (covering
//! *Donkey Kong Country 1–3*, *Banjo-Kazooie: Grunty's Revenge*, *Banjo-Pilot*,
//! *Sabre Wulf*, *It's Mr. Pants*, *Battletoads*, and *Diddy Kong Pilot*) reveals
//! exactly four distinct ARM7TDMI IWRAM mixer revisions:
//!
//! 1. [`RareAudioVersion::StandardSingleClamp`] (15 ROMs: *Donkey Kong Country* US/EU/JP,
//!    *Banjo-Kazooie: Grunty's Revenge* retail + betas, *Sabre Wulf*, *It's Mr. Pants*,
//!    *Battletoads* proto, *Diddy Kong Pilot* proto):
//!    Clean per-sample `strh`/`ldrsh` inner loop with single-sample `ldrsh` + `strb` clamping.
//! 2. [`RareAudioVersion::StandardDualHalfClamp`] (3 ROMs: *Donkey Kong Country 2* US/EU/JP):
//!    Clean per-sample `strh`/`ldrsh` inner loop with 2-sample packed `mov r4, #0x800000`
//!    `cmp`/`cmn` + `strh` clamping.
//! 3. [`RareAudioVersion::StandardDualByteClamp`] (4 ROMs: *Donkey Kong Country 3* US/EU/JP + proto):
//!    Clean per-sample `strh`/`ldrsh` inner loop with 2-sample `ldr` + `strb` x2 clamping.
//! 4. [`RareAudioVersion::PackedFastPath`] (4 ROMs: *Banjo-Pilot* US/EU + 2 protos):
//!    Lookahead-guarded 2x-packed 32-bit `str` first-voice fast path (`add r2, sl, r2, lsl #16`),
//!    which exhibits an authentic ARM negative-low-halfword `-1` carry bleed into odd samples
//!    whenever `s[2k] < 0` on the first voice, paired with a 4x-unrolled `ror #16` accumulate
//!    fast path (zero bleed), `strh` slow-path fallback on end/loop crossing, and `DualHalf` clamping.

/// Hardware DirectSound A clamp routine style used by a Rareware GBA build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RareClampMode {
    /// 1 sample/iter: `ldrsh r3, [r0], #2` -> `cmp #0x7f` / `cmn #0x80` -> `strb r3, [r1], #1`.
    SingleByteLdrsh,
    /// 2 samples/iter: `ldr r3, [r0], #4` -> `mov r4, #0x800000` `cmp`/`cmn` -> `strh r3, [r1], #2`.
    DualHalfCmn,
    /// 2 samples/iter: `ldr r3, [r0], #4` -> `lsl #16; asr #16` / `asr #16` -> `strb` x2.
    DualByteLdr32,
}

/// Distinct Rareware GBA ARM7TDMI IWRAM mixer revisions discovered across all 26 Rareware ROMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RareAudioVersion {
    /// Standard per-sample `strh` mixer + `SingleByteLdrsh` clamp (15 ROMs: *DKC 1*, *BKZ: GR*,
    /// *Sabre Wulf*, *It's Mr. Pants*, *Battletoads*, *Diddy Kong Pilot*).
    #[default]
    StandardSingleClamp,
    /// Standard per-sample `strh` mixer + `DualHalfCmn` clamp (3 ROMs: *DKC 2* US/EU/JP).
    StandardDualHalfClamp,
    /// Standard per-sample `strh` mixer + `DualByteLdr32` clamp (4 ROMs: *DKC 3* US/EU/JP + proto).
    StandardDualByteClamp,
    /// 2x-packed fast-path mixer (with authentic ARM `add r2, sl, r2, lsl #16` negative-low-halfword
    /// `-1` bleed on first voice) + `DualHalfCmn` clamp (4 ROMs: *Banjo-Pilot* US/EU + 2 protos).
    PackedFastPath,
}

/// ASCII debug string embedded in every Rareware GBA audio build.
pub const RARE_DEBUG_SIGNATURE: &[u8] = b"not enough space to put audio RAM code";

/// DirectSound A FIFO + DMA1 literal-pool triplet (`0x040000A0`, `0x040000C0`, `0x040000BC`).
pub const RARE_FIFO_DMA_SIGNATURE: [u8; 12] = [
    0xA0, 0x00, 0x00, 0x04, // REG_FIFO_A  = 0x040000A0
    0xC0, 0x00, 0x00, 0x04, // REG_DMA1DAD = 0x040000C0
    0xBC, 0x00, 0x00, 0x04, // REG_DMA1SAD = 0x040000BC
];

impl RareAudioVersion {
    /// Returns the ARM clamp loop style used by this revision.
    #[inline]
    #[must_use]
    pub const fn clamp_mode(self) -> RareClampMode {
        match self {
            Self::StandardSingleClamp => RareClampMode::SingleByteLdrsh,
            Self::StandardDualHalfClamp | Self::PackedFastPath => RareClampMode::DualHalfCmn,
            Self::StandardDualByteClamp => RareClampMode::DualByteLdr32,
        }
    }

    /// Returns `true` if this revision uses *Banjo-Pilot*'s lookahead-guarded 2x-packed
    /// 32-bit `str` first-voice fast path (exhibiting the ARM negative-low-halfword `-1` bleed).
    #[inline]
    #[must_use]
    pub const fn has_packed_first_voice_fast_path(self) -> bool {
        matches!(self, Self::PackedFastPath)
    }

    /// Maximum simultaneous hardware mixer voices for Rareware GBA builds.
    #[inline]
    #[must_use]
    pub const fn max_voices(self) -> u8 {
        8
    }

    /// Detect the exact Rareware GBA ARM mixer revision from a ROM or IWRAM image.
    ///
    /// Scans for the canonical Rareware 23-bit linear-interpolation inner loop:
    /// ```text
    /// ldrsb r1, [r0]       ; 0xE1D010D0
    /// ldrsb r2, [r0, #1]   ; 0xE1D020D1
    /// sub   r2, r2, r1     ; 0xE0422001
    /// mul   r3, r6, r2     ; 0xE0030296
    /// ```
    /// Then distinguishes `PackedFastPath` (`asr sl, r2, #7` = `0xE1A0A3C2`) vs `Standard`
    /// (`asr r2, r2, #7` = `0xE1A023C2`), and inspects the preceding clamp loop opcodes
    /// to classify `StandardSingleClamp`, `StandardDualHalfClamp`, and `StandardDualByteClamp`.
    #[must_use]
    pub fn detect_from_bytes(data: &[u8]) -> Option<Self> {
        // Canonical Rareware 23-bit linear interpolation prologue (16 bytes)
        const INTERP_SIG: [u8; 16] = [
            0xD0, 0x10, 0xD0, 0xE1, // ldrsb r1, [r0]
            0xD1, 0x20, 0xD0, 0xE1, // ldrsb r2, [r0, #1]
            0x01, 0x20, 0x42, 0xE0, // sub   r2, r2, r1
            0x96, 0x02, 0x03, 0xE0, // mul   r3, r6, r2
        ];

        let pos = data.windows(INTERP_SIG.len()).position(|w| w == INTERP_SIG)?;

        // Check `asr sl, r2, #7` (0xE1A0A3C2) vs `asr r2, r2, #7` (0xE1A023C2) at pos + 0x18
        if pos + 0x1C <= data.len() {
            let shift_word = u32::from_le_bytes([
                data[pos + 0x18],
                data[pos + 0x19],
                data[pos + 0x1A],
                data[pos + 0x1B],
            ]);
            if shift_word == 0xE1A0_A3C2 {
                return Some(Self::PackedFastPath);
            }
        }

        // Inspect clamp routine preceding the inner mixer loop (within 0x1200 bytes before pos)
        let region_start = pos.saturating_sub(0x1200);
        let region = &data[region_start..pos];

        // DualHalfCmn: `mov r4, #0x800000` (0xE3A04502) + `strh r3, [r1], #2` (0xE0C130B2)
        const MOV_R4_800000: [u8; 4] = 0xE3A0_4502u32.to_le_bytes();
        const STRH_R3_R1_2: [u8; 4] = 0xE0C1_30B2u32.to_le_bytes();
        if contains_word(region, MOV_R4_800000) && contains_word(region, STRH_R3_R1_2) {
            return Some(Self::StandardDualHalfClamp);
        }

        // DualByteLdr32: `ldr r3, [r0], #4` (0xE4903004) + `cmp lr, #0x7f` (0xE35E007F)
        const LDR_R3_R0_4: [u8; 4] = 0xE490_3004u32.to_le_bytes();
        const CMP_LR_7F: [u8; 4] = 0xE35E_007Fu32.to_le_bytes();
        if contains_word(region, LDR_R3_R0_4) && contains_word(region, CMP_LR_7F) {
            return Some(Self::StandardDualByteClamp);
        }

        Some(Self::StandardSingleClamp)
    }

    /// Quick signature check matching either Rareware's audio RAM debug string or
    /// the DirectSound A + DMA1 literal-pool triplet.
    #[must_use]
    pub fn has_rare_signature(data: &[u8]) -> bool {
        data.windows(RARE_DEBUG_SIGNATURE.len())
            .any(|w| w == RARE_DEBUG_SIGNATURE)
            || data
                .windows(RARE_FIFO_DMA_SIGNATURE.len())
                .any(|w| w == RARE_FIFO_DMA_SIGNATURE)
    }
}

#[inline]
fn contains_word(haystack: &[u8], needle: [u8; 4]) -> bool {
    haystack.windows(4).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_metadata() {
        assert_eq!(
            RareAudioVersion::StandardSingleClamp.clamp_mode(),
            RareClampMode::SingleByteLdrsh
        );
        assert!(!RareAudioVersion::StandardSingleClamp.has_packed_first_voice_fast_path());

        assert_eq!(
            RareAudioVersion::StandardDualHalfClamp.clamp_mode(),
            RareClampMode::DualHalfCmn
        );
        assert!(!RareAudioVersion::StandardDualHalfClamp.has_packed_first_voice_fast_path());

        assert_eq!(
            RareAudioVersion::StandardDualByteClamp.clamp_mode(),
            RareClampMode::DualByteLdr32
        );
        assert!(!RareAudioVersion::StandardDualByteClamp.has_packed_first_voice_fast_path());

        assert_eq!(
            RareAudioVersion::PackedFastPath.clamp_mode(),
            RareClampMode::DualHalfCmn
        );
        assert!(RareAudioVersion::PackedFastPath.has_packed_first_voice_fast_path());
        assert_eq!(RareAudioVersion::PackedFastPath.max_voices(), 8);
    }

    #[test]
    fn test_detect_all_four_versions_from_opcodes() {
        let interp: [u8; 16] = [
            0xD0, 0x10, 0xD0, 0xE1, 0xD1, 0x20, 0xD0, 0xE1, 0x01, 0x20, 0x42, 0xE0, 0x96, 0x02,
            0x03, 0xE0,
        ];

        // 1. StandardSingleClamp
        let mut buf = [0u8; 0x80];
        buf[0x40..0x50].copy_from_slice(&interp);
        buf[0x58..0x5C].copy_from_slice(&0xE1A0_23C2u32.to_le_bytes());
        assert_eq!(
            RareAudioVersion::detect_from_bytes(&buf),
            Some(RareAudioVersion::StandardSingleClamp)
        );

        // 2. StandardDualHalfClamp
        let mut dkc2 = buf;
        dkc2[0x10..0x14].copy_from_slice(&0xE3A0_4502u32.to_le_bytes());
        dkc2[0x14..0x18].copy_from_slice(&0xE0C1_30B2u32.to_le_bytes());
        assert_eq!(
            RareAudioVersion::detect_from_bytes(&dkc2),
            Some(RareAudioVersion::StandardDualHalfClamp)
        );

        // 3. StandardDualByteClamp
        let mut dkc3 = buf;
        dkc3[0x10..0x14].copy_from_slice(&0xE490_3004u32.to_le_bytes());
        dkc3[0x14..0x18].copy_from_slice(&0xE35E_007Fu32.to_le_bytes());
        assert_eq!(
            RareAudioVersion::detect_from_bytes(&dkc3),
            Some(RareAudioVersion::StandardDualByteClamp)
        );

        // 4. PackedFastPath
        let mut baje = dkc2;
        baje[0x58..0x5C].copy_from_slice(&0xE1A0_A3C2u32.to_le_bytes());
        assert_eq!(
            RareAudioVersion::detect_from_bytes(&baje),
            Some(RareAudioVersion::PackedFastPath)
        );
    }
}
