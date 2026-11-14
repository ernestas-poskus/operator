//! The `code-intelligence` binary entry point.
//!
//! The crate is a library (`src/lib.rs`) plus this thin binary. The library
//! owns the public [`code_intelligence::Operator`] facade and the modules it uses; this
//! binary dispatches the CLI, so integration tests compile the same code once
//! instead of a second copy of the binary target.

#[tokio::main]
async fn main() -> code_intelligence::error::Result<()> {
    code_intelligence::run().await
}
