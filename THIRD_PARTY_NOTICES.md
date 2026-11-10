# Third-party notices

`operator` is distributed under the MIT license; see [LICENSE](LICENSE).

This repository does not vendor third-party source code. Runtime and build
dependencies are resolved from crates.io and remain under their own licenses;
the exact versions are pinned by `Cargo.lock`. The only in-repository library is
`crates/crate-map`, which is first-party MIT code.

Notable direct dependencies include ast-grep, LanceDB, fastembed/candle,
xberg/anydoc, rmcp, and the Rust async ecosystem. Each crate's license and
copyright notice are available from its published package. No third-party
license text is redistributed here because no third-party source is vendored.
