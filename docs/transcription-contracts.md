# Transcription contract coverage

This matrix describes representative combinations rather than a Cartesian
product. Routing assertions establish which request reaches the engine;
behavioral tests establish output, progress and lifecycle effects. A passing
mock cannot establish recognition quality or native numerical equivalence.

| Source / channel mode | VAD / diarization | Overrides | Progress / cancellation | Transport | Contract coverage |
|---|---|---|---|---|---|
| Encoded WAV and Opus; mono, dual-mono, genuine stereo, multichannel fallback | VAD enabled/disabled; split speaker labels take precedence over diarization | Punctuation, ITN, VAD, hotwords and duration budget preserved | Same abort, snapshot and progress handles reach every route | Shared REST / jobs preparation | `test_channel_routing_preserves_request_context` in [file_transcribe.rs](../crates/gigastt/src/server/file_transcribe.rs), 120 cases |
| Mono WAV; dual-mono and genuine stereo Opus | Explicit VAD off; no offline diarization | ITN off overrides boot ITN on with a number-word output sentinel; punctuation/hotwords explicitly off | Successful job reaches audio-duration completion and one Done event | REST versus production job executor | `test_rest_and_job_split_modes_preserve_observable_request_override` in [transcription_contracts.rs](../crates/gigastt/src/server/http/tests/transcription_contracts.rs); compares word text, timestamps, confidence and speaker labels |
| Buffered channels and container channel streams | VAD off | Engine defaults | Cumulative sample work across channels; interrupted second channel preserves earlier text | Core API | Channel progress cases in [paths.rs](../crates/gigastt-core/src/inference/engine/tests/paths.rs) and channel cancellation in [cancellation.rs](../crates/gigastt-core/src/inference/engine/tests/cancellation.rs) |
| Split-channel scan and whole-buffer decode | VAD preparation; mono fallback and Opus reuse | Existing duration ceilings | Cancellation before probing and between packets/blocks | Core API / shared preparation | [Audio cancellation tests](../crates/gigastt-core/src/inference/audio/tests/cancellation.rs) and `test_opus_preparation_*` in [stream.rs](../crates/gigastt-core/src/inference/audio/tests/stream.rs) |
| Windowed file and predecoded samples | Streaming and buffered VAD; lazy speaker loading; offline diarization | ITN and punctuation stages | Cancel before pulling a source, before speaker load, between VAD/embedding calls, and during postprocessing | Core API | [Engine cancellation tests](../crates/gigastt-core/src/inference/engine/tests/cancellation.rs), [diarization adapter tests](../crates/gigastt-core/src/inference/diarization.rs), and existing VAD cancellation tests |
| Healthy multi-window / multichannel work | Independent of recognition backend | Per-request progress context | Completed work resets deadline; channel transition cannot reset sample counter | REST / jobs watchdog | `test_watchdog_channel_transition_preserves_liveness`, early-progress/stall tests in [file_transcribe.rs](../crates/gigastt/src/server/file_transcribe.rs) |
| Encoded file with blocked native inference or unread output | Streaming recognizer | Transport-specific commitment | Timeout ends response before worker exit; output backpressure has a separate bound; reservation remains owned | Native SSE / OpenAI SSE | [stream_backpressure.rs](../crates/gigastt/src/server/http/tests/stream_backpressure.rs) and paused-time [watchdog phase tests](../crates/gigastt/src/server/http/stream_watchdog.rs) |
| Sub-stride final audio | Streaming recognizer | Existing commitment policies | Encoder/joiner error or panic cannot become terminal success | WS, SSE, OpenAI and bindings | [stream_finalization.rs](../crates/gigastt/src/server/http/tests/stream_finalization.rs), core finalization tests and binding flush regressions |
| Queued/processing job and terminal subscription | Independent of recognition backend | Existing job request | Subscribe before/after transition, concurrent cancel/complete and admission races yield one consistent terminal outcome | Job store / job SSE | [Job event tests](../crates/gigastt/src/server/jobs/tests/events.rs) and [HTTP job tests](../crates/gigastt/src/server/http/tests/jobs.rs) |
| Cancellation wins before executor registers abort | Encoded genuine stereo Opus | Split request | Real executor must not enter armed inference; late completion rejected, subscriber receives only Cancelled and pool recovers | Production job executor / store | `test_cancel_before_executor_registration_preserves_terminal_subscription` in [transcription_contracts.rs](../crates/gigastt/src/server/http/tests/transcription_contracts.rs) |
| Short speech as Path, Bytes, Samples; genuine stereo as Channels and ChannelStreams | VAD off; no offline diarization | Explicit ITN/punctuation off | Exact final sample-work totals | Core API, real INT8 RNN-T | `test_offline_source_contracts_preserve_numerical_output` in [e2e_rest.rs](../crates/gigastt/tests/e2e_rest.rs), five requests comparing full serialized numerical output |

The model-backed row uses existing Golos fixtures and requires all four RNN-T
bundle files. It fails when explicitly selected without them and never downloads
weights itself. It does not assert equality between offline and incremental
streaming transcripts, nor does it establish diarization quality or real-model
VAD equivalence. Those are separate quality/fixture requirements. Cancellation
boundaries and unavoidable synchronous intervals are documented in the
[runbook](runbook.md#cancellation-boundaries-and-resource-ownership).

## Running the additional checks

The two model-free checks are included in the normal workspace unit suite:

```sh
cargo test -p gigastt --lib transcription_contracts
```

The numerical parity check is included in the existing serial main-branch
`e2e_rest` job. Its exact standalone selector has a nonzero preflight:

```sh
cargo test -p gigastt --test e2e_rest -- --ignored --list --exact test_offline_source_contracts_preserve_numerical_output | grep -Fx 'test_offline_source_contracts_preserve_numerical_output: test'
cargo test -p gigastt --test e2e_rest -- --ignored --exact test_offline_source_contracts_preserve_numerical_output --test-threads=1 --nocapture
```

Use `--workspace --lib --bins` for focused unit checks; the long WER benchmark
requires explicit `--run`.
The new checks reuse existing dependencies, fixtures and the cached RNN-T model;
they add no model download or extra CI job. Compilation and model loading must
be distinguished from warmed test execution when comparing timings.

## Incremental CI budget

The budget for these additions is 15 seconds in the existing unit job and
60 seconds in the existing serial model job, excluding compilation and model
cache restoration. The existing job timeout remains the hard CI bound; these
are runtime budgets to investigate if exceeded, not guarantees about arbitrary
hardware or native-call interruption.

A Linux x86-64 debug build measured 5.43 seconds for the two model-free tests
and 11.16 seconds for the one model-backed test (11.13 seconds including engine
load inside the test). The model test made five API requests through one loaded
engine, with seven mono/channel recognition passes. The selector preflight
listed exactly one test. These are local measurements, not CI-runner timings
or whole-application performance claims.

The mock suite was checked against two temporary mutations: dropping request
overrides made the raw number-word assertion fail, and omitting cancellation
seeding made the executor return the armed inference failure instead of
`Cancelled`. Both mutations were removed before final checks. Existing
watchdog and atomic event tests provide the complementary false-timeout and
contradictory-terminal-event regressions.
