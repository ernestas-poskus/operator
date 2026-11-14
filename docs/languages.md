# Languages

Detection uses `ast_grep_language::SupportLang::from_path`; support is a
capability contract, not a hand-written extension table.

| Language | AST search | Index | Outline |
|---|---|---|---|
| Bash, C, C++, C#, CSS, Dart, Elixir, Go, Haskell, HCL, HTML, Java, JavaScript, JSON, Kotlin, Lua, Markdown, Nix, PHP, Python, Ruby, Rust, Scala, Solidity, Swift, TSX, TypeScript, YAML, Zig | yes | yes | bundled where ast-grep ships a rule; generic file entry otherwise |

Dart is included in the ast-grep `builtin-parser` feature set.

`code-intelligence init` detects and intersects; only selected languages are indexed or
searchable. There is no LSP process and no compiler-grade reference/type
semantics in code-intelligence.
