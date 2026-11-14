# code-intelligence developer targets.
.DEFAULT_GOAL := help
CARGO ?= cargo

.PHONY: help setup verify install check check-lib test fmt clippy doc

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

setup:
	$(CARGO) fetch --locked

check:
	$(CARGO) check --locked

check-lib:
	$(CARGO) check --locked --no-default-features --all-targets

test:
	$(CARGO) nextest run --status-level fail -p code-intelligence

fmt:
	$(CARGO) fmt -p code-intelligence

clippy:
	$(CARGO) clippy --all-targets

doc:
	$(CARGO) doc --no-deps

verify: check check-lib test fmt clippy ## Run the source gates

install:
	$(CARGO) install --path . --force --locked
