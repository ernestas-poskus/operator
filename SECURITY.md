# Security Policy

## Supported versions

Security fixes land on the latest 0.1.x release. Older releases and unreleased
builds are not supported.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Do not open a public issue for a security problem. Report it privately through
GitHub Security Advisories:

<https://github.com/ernestas-poskus/operator/security/advisories/new>

Include the affected version or commit, the impact, reproduction steps, and any
suggested fix. The maintainer will confirm receipt, investigate, and coordinate
the fix and disclosure date with you. Do not include live API keys or other
secrets in a report; rotate any key that may have been exposed.

## Scope

Reports are most useful when they touch the trust boundaries below.

- **Workspace path containment.** `read_source` or index discovery following
  symlinks, `..`, or absolute paths outside the workspace.
- **MCP stdio exposure.** Tool arguments that reach a shell, file write, or
  network path; malformed patterns/rules that cause unbounded CPU or memory
  use.
- **Journal privacy.** Raw source content, absolute paths, or credentials
  leaking into the `learn` journal; journal files readable by other users.
- **Index artifacts.** Old index tables are not migrated and must not be read
  as trusted input.

## Secret handling

The binary embeds no API keys. Retrieval runs in-process with fastembed; the
only outbound request is the one-time Hugging Face model download. No
retrieval bearer token exists in the current implementation.

## Disclosure

Fixes are prepared privately, released, and then described in a GitHub Security
Advisory. Reporters are credited unless they ask to stay anonymous.
