//! Newtypes over the primitive values that cross module and API boundaries.
//!
//! Every value with domain meaning is a distinct type instead of a bare
//! `String`, integer, float, path or boolean: a language id can never be
//! mistaken for a repo path, a line number for a byte count, or a token limit
//! for a page size. Each wrapper is `#[serde(transparent)]` (and
//! `#[schemars(transparent)]`), so the JSON and `.operator.toml` wire formats
//! are unchanged — the typing is compile-time only.
//!
//! Convention: the wrapper name is the meaning (`LineNumber`, `LanguageId`),
//! `new`/`get` convert explicitly, `From`/`Into` convert implicitly, and the
//! wrappers implement the arithmetic and comparison operators against their
//! inner primitive so call sites stay readable. CLI switches (`argh`) remain
//! plain `bool` because `argh` requires booleans for `#[argh(switch)]`; the
//! parsed result is converted to the named flag at the boundary.

use std::borrow::Borrow;
use std::fmt;
use std::ops::{Add, AddAssign, Deref, Div, Mul, Neg, Sub, SubAssign};

use serde::{Deserialize, Serialize};

/// Define a newtype over `String` with the usual conversions and comparisons.
macro_rules! string_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug,
            Clone,
            Default,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        #[schemars(transparent)]
        pub struct $name(String);

        impl $name {
            /// Wrap a string without validation.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Borrow the wrapped string.
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Unwrap the string.
            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl Deref for $name {
            type Target = str;

            fn deref(&self) -> &str {
                &self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl Borrow<str> for $name {
            fn borrow(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl From<&String> for $name {
            fn from(value: &String) -> Self {
                Self(value.clone())
            }
        }

        impl From<&$name> for String {
            fn from(value: &$name) -> String {
                value.0.clone()
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> String {
                value.0
            }
        }

        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.0 == other
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.0 == *other
            }
        }

        impl PartialEq<String> for $name {
            fn eq(&self, other: &String) -> bool {
                &self.0 == other
            }
        }

        impl PartialEq<$name> for str {
            fn eq(&self, other: &$name) -> bool {
                self == other.0
            }
        }

        impl PartialEq<$name> for &str {
            fn eq(&self, other: &$name) -> bool {
                *self == other.0
            }
        }

        impl PartialEq<$name> for String {
            fn eq(&self, other: &$name) -> bool {
                self == &other.0
            }
        }

        impl std::str::FromStr for $name {
            type Err = std::convert::Infallible;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Ok(Self(value.to_string()))
            }
        }
    };
}

/// Define a newtype over an integer with arithmetic against the inner type.
macro_rules! integer_newtype {
    ($(#[$meta:meta])* $name:ident, $inner:ty) => {
        $(#[$meta])*
        #[derive(
            Debug,
            Clone,
            Copy,
            Default,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        #[schemars(transparent)]
        pub struct $name($inner);

        impl $name {
            /// Wrap an integer.
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            /// Unwrap the integer.
            pub const fn get(self) -> $inner {
                self.0
            }

            /// Whether the value is zero.
            pub const fn is_zero(self) -> bool {
                self.0 == 0
            }

            /// Keep the smaller of the two values.
            pub fn min(self, other: $inner) -> Self {
                Self(self.0.min(other))
            }

            /// Keep the larger of the two values.
            pub fn max(self, other: $inner) -> Self {
                Self(self.0.max(other))
            }

            /// Constrain the value to `min..=max`.
            pub fn clamp_to(self, min: $inner, max: $inner) -> Self {
                Self(self.0.clamp(min, max))
            }

            /// Saturating subtraction, staying inside the wrapper.
            pub fn saturating_sub(self, other: $inner) -> Self {
                Self(self.0.saturating_sub(other))
            }

            /// Saturating addition, staying inside the wrapper.
            pub fn saturating_add(self, other: $inner) -> Self {
                Self(self.0.saturating_add(other))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl From<$inner> for $name {
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> $inner {
                value.0
            }
        }

        impl PartialEq<$inner> for $name {
            fn eq(&self, other: &$inner) -> bool {
                self.0 == *other
            }
        }

        impl PartialOrd<$inner> for $name {
            fn partial_cmp(&self, other: &$inner) -> Option<std::cmp::Ordering> {
                self.0.partial_cmp(other)
            }
        }

        impl PartialEq<$name> for $inner {
            fn eq(&self, other: &$name) -> bool {
                *self == other.0
            }
        }

        impl PartialOrd<$name> for $inner {
            fn partial_cmp(&self, other: &$name) -> Option<std::cmp::Ordering> {
                self.partial_cmp(&other.0)
            }
        }

        impl Add<$inner> for $name {
            type Output = $name;

            fn add(self, rhs: $inner) -> Self::Output {
                Self(self.0 + rhs)
            }
        }

        impl Add<$name> for $name {
            type Output = $name;

            fn add(self, rhs: $name) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }

        impl Add<$name> for $inner {
            type Output = $name;

            fn add(self, rhs: $name) -> Self::Output {
                $name(self + rhs.0)
            }
        }

        impl Sub<$inner> for $name {
            type Output = $name;

            fn sub(self, rhs: $inner) -> Self::Output {
                Self(self.0 - rhs)
            }
        }

        impl Sub<$name> for $name {
            type Output = $name;

            fn sub(self, rhs: $name) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }

        impl Mul<$inner> for $name {
            type Output = $name;

            fn mul(self, rhs: $inner) -> Self::Output {
                Self(self.0 * rhs)
            }
        }

        impl Div<$inner> for $name {
            type Output = $name;

            fn div(self, rhs: $inner) -> Self::Output {
                Self(self.0 / rhs)
            }
        }

        impl AddAssign<$inner> for $name {
            fn add_assign(&mut self, rhs: $inner) {
                self.0 += rhs;
            }
        }

        impl AddAssign<$name> for $name {
            fn add_assign(&mut self, rhs: $name) {
                self.0 += rhs.0;
            }
        }

        impl SubAssign<$inner> for $name {
            fn sub_assign(&mut self, rhs: $inner) {
                self.0 -= rhs;
            }
        }

        impl std::iter::Sum for $name {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                Self(iter.map(|value| value.0).sum())
            }
        }

        impl<'a> std::iter::Sum<&'a $name> for $name {
            fn sum<I: Iterator<Item = &'a $name>>(iter: I) -> Self {
                Self(iter.map(|value| value.0).sum())
            }
        }

        impl std::str::FromStr for $name {
            type Err = <$inner as std::str::FromStr>::Err;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                value.parse::<$inner>().map(Self)
            }
        }
    };
}

/// Define a newtype over `f32` with arithmetic against the inner type.
macro_rules! float_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug,
            Clone,
            Copy,
            Default,
            PartialEq,
            PartialOrd,
            serde::Serialize,
            serde::Deserialize,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        #[schemars(transparent)]
        pub struct $name(f32);

        impl $name {
            /// Wrap a float.
            pub const fn new(value: f32) -> Self {
                Self(value)
            }

            /// Unwrap the float.
            pub const fn get(self) -> f32 {
                self.0
            }

            /// Constrain the value to `min..=max`.
            pub fn clamp_to(self, min: f32, max: f32) -> Self {
                Self(self.0.clamp(min, max))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl From<f32> for $name {
            fn from(value: f32) -> Self {
                Self(value)
            }
        }

        impl From<$name> for f32 {
            fn from(value: $name) -> f32 {
                value.0
            }
        }

        impl PartialEq<f32> for $name {
            fn eq(&self, other: &f32) -> bool {
                self.0 == *other
            }
        }

        impl PartialOrd<f32> for $name {
            fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
                self.0.partial_cmp(other)
            }
        }

        impl PartialEq<$name> for f32 {
            fn eq(&self, other: &$name) -> bool {
                *self == other.0
            }
        }

        impl PartialOrd<$name> for f32 {
            fn partial_cmp(&self, other: &$name) -> Option<std::cmp::Ordering> {
                self.partial_cmp(&other.0)
            }
        }

        impl Add<$name> for $name {
            type Output = $name;

            fn add(self, rhs: $name) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }

        impl Sub<$name> for $name {
            type Output = $name;

            fn sub(self, rhs: $name) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }

        impl Mul<f32> for $name {
            type Output = $name;

            fn mul(self, rhs: f32) -> Self::Output {
                Self(self.0 * rhs)
            }
        }

        impl Div<f32> for $name {
            type Output = $name;

            fn div(self, rhs: f32) -> Self::Output {
                Self(self.0 / rhs)
            }
        }

        impl Neg for $name {
            type Output = $name;

            fn neg(self) -> Self::Output {
                Self(-self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = std::num::ParseFloatError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                value.parse::<f32>().map(Self)
            }
        }
    };
}

/// Define a newtype over `PathBuf`.
macro_rules! path_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug,
            Clone,
            Default,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        #[schemars(transparent)]
        pub struct $name(std::path::PathBuf);

        impl $name {
            /// Wrap a path.
            pub fn new(value: impl Into<std::path::PathBuf>) -> Self {
                Self(value.into())
            }

            /// Borrow the path.
            pub fn as_path(&self) -> &std::path::Path {
                &self.0
            }

            /// Unwrap the path.
            pub fn into_path_buf(self) -> std::path::PathBuf {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.display().fmt(f)
            }
        }

        impl Deref for $name {
            type Target = std::path::Path;

            fn deref(&self) -> &std::path::Path {
                &self.0
            }
        }

        impl AsRef<std::path::Path> for $name {
            fn as_ref(&self) -> &std::path::Path {
                &self.0
            }
        }

        impl Borrow<std::path::Path> for $name {
            fn borrow(&self) -> &std::path::Path {
                &self.0
            }
        }

        impl From<std::path::PathBuf> for $name {
            fn from(value: std::path::PathBuf) -> Self {
                Self(value)
            }
        }

        impl From<&std::path::Path> for $name {
            fn from(value: &std::path::Path) -> Self {
                Self(value.to_path_buf())
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(std::path::PathBuf::from(value))
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(std::path::PathBuf::from(value))
            }
        }

        impl From<$name> for std::path::PathBuf {
            fn from(value: $name) -> std::path::PathBuf {
                value.0
            }
        }

        impl PartialEq<std::path::Path> for $name {
            fn eq(&self, other: &std::path::Path) -> bool {
                self.0 == other
            }
        }

        impl std::str::FromStr for $name {
            type Err = std::convert::Infallible;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Ok(Self(std::path::PathBuf::from(value)))
            }
        }
    };
}

/// Define a newtype over `bool` for a named flag.
macro_rules! flag_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug,
            Clone,
            Copy,
            Default,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        #[schemars(transparent)]
        pub struct $name(bool);

        impl $name {
            /// Wrap a boolean.
            pub const fn new(value: bool) -> Self {
                Self(value)
            }

            /// A set flag.
            pub const fn on() -> Self {
                Self(true)
            }

            /// A cleared flag.
            pub const fn off() -> Self {
                Self(false)
            }

            /// Unwrap the boolean.
            pub const fn get(self) -> bool {
                self.0
            }

            /// Whether the flag is set.
            pub const fn is_on(self) -> bool {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl From<bool> for $name {
            fn from(value: bool) -> Self {
                Self(value)
            }
        }

        impl From<$name> for bool {
            fn from(value: $name) -> bool {
                value.0
            }
        }

        impl PartialEq<bool> for $name {
            fn eq(&self, other: &bool) -> bool {
                self.0 == *other
            }
        }

        impl PartialEq<$name> for bool {
            fn eq(&self, other: &$name) -> bool {
                *self == other.0
            }
        }

        impl std::ops::Not for $name {
            type Output = $name;

            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }
    };
}

// ── Language and workspace identity ─────────────────────────────────────────

/// A canonical ast-grep language id (`rust`, `typescript`, …).
///
/// Construction goes through the catalog: [`LanguageId::from_lang`] for a
/// known [`ast_grep_language::SupportLang`], [`LanguageId::parse`] for raw
/// text. Deserialization validates for the same reason, so a malformed
/// `.operator.toml` fails loudly at load instead of at first search.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, schemars::JsonSchema,
)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct LanguageId(String);

impl LanguageId {
    /// Canonical id for a catalog language.
    pub fn from_lang(lang: ast_grep_language::SupportLang) -> Self {
        Self(crate::languages::canonical_id(lang).to_string())
    }

    /// Parse a canonical id or a catalog alias.
    pub fn parse(raw: &str) -> Option<Self> {
        crate::languages::parse_id(raw).map(Self::from_lang)
    }

    /// Borrow the id.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Unwrap the id.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for LanguageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Deref for LanguageId {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for LanguageId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for LanguageId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<ast_grep_language::SupportLang> for LanguageId {
    fn from(lang: ast_grep_language::SupportLang) -> Self {
        Self::from_lang(lang)
    }
}

impl From<String> for LanguageId {
    fn from(value: String) -> Self {
        Self::parse(&value).unwrap_or(Self(value))
    }
}

impl From<&str> for LanguageId {
    fn from(value: &str) -> Self {
        Self::from(value.to_string())
    }
}

impl From<&LanguageId> for String {
    fn from(value: &LanguageId) -> String {
        value.0.clone()
    }
}

impl From<LanguageId> for String {
    fn from(value: LanguageId) -> String {
        value.0
    }
}

impl PartialEq<str> for LanguageId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for LanguageId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<String> for LanguageId {
    fn eq(&self, other: &String) -> bool {
        &self.0 == other
    }
}

impl PartialEq<LanguageId> for str {
    fn eq(&self, other: &LanguageId) -> bool {
        self == other.0
    }
}

impl PartialEq<LanguageId> for &str {
    fn eq(&self, other: &LanguageId) -> bool {
        *self == other.0
    }
}

impl PartialEq<LanguageId> for String {
    fn eq(&self, other: &LanguageId) -> bool {
        self == &other.0
    }
}

impl std::str::FromStr for LanguageId {
    type Err = crate::error::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
            .ok_or_else(|| crate::error::Error::Config(format!("unsupported language `{value}`")))
    }
}

impl<'de> Deserialize<'de> for LanguageId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw)
            .ok_or_else(|| serde::de::Error::custom(format!("unsupported language `{raw}`")))
    }
}

string_newtype!(
    #[doc = "Human-readable language name (`TypeScript`)."]
    LanguageName
);

// ── Source identity ─────────────────────────────────────────────────────────

string_newtype!(
    #[doc = "A repository-relative source path, using `/` separators."]
    RepoPath
);
string_newtype!(
    #[doc = "Source text: a file body, chunk body, snippet or matched line."]
    SourceText
);
string_newtype!(
    #[doc = "One outline symbol's name."]
    SymbolName
);
string_newtype!(
    #[doc = "One outline symbol's kind (`function`, `class`, …)."]
    SymbolKind
);
string_newtype!(
    #[doc = "One outline symbol's signature line."]
    Signature
);
string_newtype!(
    #[doc = "A stable chunk identity (`<path>#<index>`)."]
    ChunkId
);
string_newtype!(
    #[doc = "A content hash of a source file."]
    FileHash
);
string_newtype!(
    #[doc = "The crate a source file belongs to."]
    CrateName
);

path_newtype!(
    #[doc = "The workspace root a [`crate::config::Config`] operates on."]
    WorkspaceRoot
);
path_newtype!(
    #[doc = "Retrieval index/cache directory."]
    CacheDir
);
path_newtype!(
    #[doc = "Retrieval manifest file."]
    ManifestPath
);
path_newtype!(
    #[doc = "Downloaded model snapshot directory."]
    ModelCacheDir
);
path_newtype!(
    #[doc = "Optional JSONL query log."]
    QueryLogPath
);
path_newtype!(
    #[doc = "Local learn-journal directory."]
    JournalDir
);
path_newtype!(
    #[doc = "The workspace configuration file."]
    ConfigPath
);

// ── Search input ────────────────────────────────────────────────────────────

string_newtype!(
    #[doc = "A user-visible query string."]
    QueryText
);
string_newtype!(
    #[doc = "Optional `path` prefix filter (substring match, as before)."]
    PathFilter
);
string_newtype!(
    #[doc = "Comma-separated `paths` filter for structural search."]
    PathsFilter
);
string_newtype!(
    #[doc = "An ast-grep pattern."]
    PatternText
);
string_newtype!(
    #[doc = "An ast-grep YAML rule."]
    RuleText
);

integer_newtype!(
    #[doc = "Maximum number of results a tool returns."]
    Limit,
    usize
);
integer_newtype!(
    #[doc = "Files counted by detection, indexing or freshness."]
    FileCount,
    usize
);
integer_newtype!(
    #[doc = "Chunk rows counted in an index or refresh."]
    ChunkCount,
    usize
);
integer_newtype!(
    #[doc = "Table rows in the retrieval index."]
    RowCount,
    usize
);
integer_newtype!(
    #[doc = "A 1-based line number."]
    LineNumber,
    u32
);
integer_newtype!(
    #[doc = "A 1-based column number."]
    ColumnNumber,
    u32
);
integer_newtype!(
    #[doc = "A zero-based chunk ordinal within a file."]
    ChunkIndex,
    u32
);
integer_newtype!(
    #[doc = "A maximum token budget."]
    TokenLimit,
    usize
);
integer_newtype!(
    #[doc = "A candidate-pool size."]
    CandidateCount,
    usize
);
integer_newtype!(
    #[doc = "An embedding width."]
    EmbedDim,
    usize
);
integer_newtype!(
    #[doc = "Lines per chunk."]
    ChunkLines,
    usize
);
integer_newtype!(
    #[doc = "Lines repeated between chunk windows."]
    ChunkOverlap,
    usize
);
integer_newtype!(
    #[doc = "Bytes returned by a tool call."]
    ResultBytes,
    usize
);
integer_newtype!(
    #[doc = "Results returned by a tool call."]
    ResultCount,
    usize
);
integer_newtype!(
    #[doc = "Journaled tool calls."]
    CallCount,
    usize
);
integer_newtype!(
    #[doc = "Journaled tool errors."]
    ErrorCount,
    usize
);
integer_newtype!(
    #[doc = "Learn signals found."]
    SignalCount,
    usize
);
integer_newtype!(
    #[doc = "A position within a journal window."]
    PositionIndex,
    usize
);
integer_newtype!(
    #[doc = "The `.operator.toml` schema version."]
    SchemaVersion,
    u32
);
integer_newtype!(
    #[doc = "Pages in a document."]
    PageCount,
    u32
);
integer_newtype!(
    #[doc = "A wall-clock timestamp in Unix milliseconds."]
    UnixMillis,
    u64
);
integer_newtype!(
    #[doc = "A Unix-seconds timestamp used in host registration files."]
    UpdatedAt,
    i64
);
integer_newtype!(
    #[doc = "An elapsed duration in milliseconds."]
    DurationMs,
    u64
);
float_newtype!(
    #[doc = "A retrieval/relevance score."]
    Score
);
float_newtype!(
    #[doc = "The dense/lexical fusion weight."]
    Weight
);
float_newtype!(
    #[doc = "The cross-encoder/fused blend factor."]
    Blend
);

// ── Retrieval configuration ─────────────────────────────────────────────────

string_newtype!(
    #[doc = "Embedding/rerank model repository or name."]
    ModelName
);
string_newtype!(
    #[doc = "Retrieval table name."]
    TableName
);
string_newtype!(
    #[doc = "Query prefix prepended before embedding."]
    QueryPrefix
);
string_newtype!(
    #[doc = "Instruction prepended to rerank pairs."]
    Instruction
);
string_newtype!(
    #[doc = "Workspace-relative include path."]
    IncludePath
);
string_newtype!(
    #[doc = "Workspace-relative exclude path."]
    ExcludePath
);

flag_newtype!(RerankEnabled);
flag_newtype!(RerankContext);
flag_newtype!(FusionGate);
flag_newtype!(DualRoute);
flag_newtype!(QueryExpansion);
flag_newtype!(QueryRouting);
flag_newtype!(Documents);
flag_newtype!(EnrichPath);
flag_newtype!(EnrichModule);
flag_newtype!(EnrichSymbols);
flag_newtype!(FtsDegraded);

// ── Learn journal ───────────────────────────────────────────────────────────

string_newtype!(
    #[doc = "A tool name as exposed to hosts (`search_code`, …)."]
    ToolName
);
string_newtype!(
    #[doc = "A journaled process/session id."]
    SessionId
);
string_newtype!(
    #[doc = "A hash of the redacted tool arguments."]
    ArgsHash
);
string_newtype!(
    #[doc = "The structural shape of the tool arguments."]
    ArgsShape
);
string_newtype!(
    #[doc = "A human-readable learn suggestion."]
    Suggestion
);
string_newtype!(
    #[doc = "The kind of learn signal."]
    SignalKind
);
string_newtype!(
    #[doc = "An error message kept in a journal record or tool result."]
    ErrorMessage
);
flag_newtype!(
    #[doc = "Whether a tool call succeeded."]
    Success
);

// ── Documents and hosts ─────────────────────────────────────────────────────

string_newtype!(
    #[doc = "A document format (`pdf`, `docx`, …)."]
    DocumentFormat
);
string_newtype!(
    #[doc = "One host registration file."]
    HostFile
);
string_newtype!(
    #[doc = "One command-line argument of a host registration transport."]
    CommandArg
);

/// A collection of transports/host argv entries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Argv(Vec<CommandArg>);

impl Argv {
    /// Wrap arguments.
    pub fn new(args: Vec<CommandArg>) -> Self {
        Self(args)
    }

    /// Borrow the arguments.
    pub fn as_slice(&self) -> &[CommandArg] {
        &self.0
    }

    /// Unwrap the arguments.
    pub fn into_inner(self) -> Vec<CommandArg> {
        self.0
    }
}

impl Deref for Argv {
    type Target = [CommandArg];

    fn deref(&self) -> &[CommandArg] {
        &self.0
    }
}

impl From<Vec<CommandArg>> for Argv {
    fn from(value: Vec<CommandArg>) -> Self {
        Self(value)
    }
}

impl From<Argv> for Vec<CommandArg> {
    fn from(value: Argv) -> Vec<CommandArg> {
        value.0
    }
}

impl<'a> IntoIterator for &'a Argv {
    type Item = &'a CommandArg;
    type IntoIter = std::slice::Iter<'a, CommandArg>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// A dense embedding vector.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Wrap a vector.
    pub fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    /// Borrow the values.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Unwrap the values.
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Deref for Embedding {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.0
    }
}

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(value: Vec<f32>) -> Self {
        Self(value)
    }
}

impl From<Embedding> for Vec<f32> {
    fn from(value: Embedding) -> Vec<f32> {
        value.0
    }
}
