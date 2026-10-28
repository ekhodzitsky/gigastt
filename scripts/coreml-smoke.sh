#!/bin/bash
set -euo pipefail

binary=${1:-./target/release/gigastt}
fixture=${2:-crates/gigastt/tests/fixtures/golos_00.wav}
status=0
RUST_LOG=warn "$binary" transcribe "$fixture" >transcript.txt 2>stderr.log || status=$?
echo '--- stderr ---'
cat stderr.log
echo '--- transcript ---'
cat transcript.txt
if [ "$status" -ne 0 ]; then
    echo "::error::CoreML inference failed (exit $status)"
    # Diagnostic control only: CPU success must never make the CoreML gate pass.
    RUST_LOG=warn "$binary" transcribe "$fixture" --execution-provider cpu >cpu-transcript.txt 2>cpu-stderr.log || true
    echo '--- CPU control stderr ---'
    cat cpu-stderr.log
    echo '--- CPU control transcript ---'
    cat cpu-transcript.txt
    exit 1
fi
if grep -q 'falling back to CPU execution provider' stderr.log; then
    echo '::error::CoreML EP fell back to CPU at runtime'
    exit 1
fi
if ! grep -q 'тенге' transcript.txt; then
    echo '::error::Transcript does not contain the expected reference text'
    exit 1
fi
