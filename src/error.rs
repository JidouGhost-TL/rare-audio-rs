//! Error types for Rareware GBA Sound Engine parsing and synthesis.

use core::fmt;

/// Errors returned when parsing Rareware GBA sample descriptors, song headers, or audio streams.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer ended before a complete descriptor or payload could be read.
    UnexpectedEof,
    /// Sample descriptor or song header field failed structural validation.
    InvalidHeader(&'static str),
    /// Loop length exceeded the sample payload length.
    InvalidLoopRange {
        /// Declared loop length in bytes.
        loop_length: u32,
        /// Total sample length in bytes.
        pcm_length: usize,
    },
    /// Playback mode was outside the valid `1..=4` range.
    InvalidPlaybackMode(u32),
    /// Sample index was not found in the active bank.
    InvalidSample(usize),
    /// Corrupted or unsupported Rareware stream data.
    CorruptedData(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => {
                write!(f, "Unexpected end of data while parsing Rareware audio")
            }
            Self::InvalidHeader(msg) => write!(f, "Invalid Rareware header: {msg}"),
            Self::InvalidLoopRange {
                loop_length,
                pcm_length,
            } => write!(
                f,
                "Invalid Rareware loop length {loop_length} for {pcm_length}-byte sample"
            ),
            Self::InvalidPlaybackMode(mode) => {
                write!(f, "Invalid Rareware playback mode {mode} (expected 1..=4)")
            }
            Self::InvalidSample(id) => write!(f, "Invalid Rareware sample index {id}"),
            Self::CorruptedData(msg) => write!(f, "Corrupted Rareware data: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience result alias for `rare-audio`.
pub type Result<T> = core::result::Result<T, Error>;
