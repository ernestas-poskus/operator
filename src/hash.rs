//! Stable hexadecimal rendering for SHA-256 digests.
//!
//! sha2 0.11 dropped the `LowerHex` formatting impl on its digest arrays; one
//! shared helper keeps every content key, cache key and prompt-prefix hash in
//! the same lowercase two-hex-digits-per-byte form.

use std::fmt::Write as _;

/// Render a digest as lowercase hex, two characters per byte.
pub(crate) fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing to a String cannot fail.
        let _ = write!(out, "{byte:02x}");
    }
    out
}
