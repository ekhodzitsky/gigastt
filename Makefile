.PHONY: check fix

check:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	# Focus on model-free unit and binary tests. Matches .githooks/pre-commit.
	cargo test --workspace --lib --bins

fix:
	cargo fmt --all
	cargo clippy --workspace --all-targets --fix -- -D warnings
