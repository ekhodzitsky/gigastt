# Privacy and data flow

gigastt performs recognition on the host running the engine. It does not send
audio or transcripts to a third-party recognition service and has no telemetry
or analytics uploader. The deployment still determines where data travels:
when a remote client calls your server, audio crosses that connection and the
transcript is returned to that client.

## Processing and storage

- CLI and in-process bindings read the audio you supply and return recognized
  text to the caller. CLI `--output`, batch and watch commands can write results
  to the selected destination.
- HTTP/WebSocket uploads and active recognition buffers are held in memory.
  Optional `/v1/jobs` keeps job data/results in memory under configured limits
  and TTL; it is not a durable disk queue. See [long recordings](long-recordings.md).
- Application tracing omits recognized transcript text. Logs can include paths,
  peer addresses, errors, timings and request metadata. A caller, reverse proxy,
  shell redirection or application wrapper can separately record requests or
  transcripts; those components are outside gigastt's logging policy.
- Model weights and optimized graph caches are stored on disk. Removing a job
  or completing transcription does not remove those reusable model assets.

## Network activity

The built-in downloader fetches missing models from the sources in the
[model catalog](models.md): GitHub Releases for RNN-T and Silero VAD, and
Hugging Face for multilingual ASR, punctuation and WeSpeaker. Downloads verify
pinned SHA-256 hashes and use atomic replacement. Provisioning these assets
requires network access and exposes the usual connection metadata to their hosts;
it does not upload recognition input.

After provisioning the required assets, use `--offline` or `GIGASTT_OFFLINE=1`
to prohibit model downloads. Missing required ASR files cause an error;
optional punctuation/VAD setup can warn and continue without that processor.
Bindings load side-provisioned files rather than downloading them implicitly.

Building from source is a separate network boundary. Cargo resolves/downloads
dependencies and the default ORT build fetches a checksummed native library.
An offline build must provision both Rust dependencies and native inputs; see
[embedding and linking](embedding-packaging.md).

## Server access

The default listener is `127.0.0.1:9876`. Non-loopback binding requires
`--bind-all` or `GIGASTT_ALLOW_BIND_ANY=1`. The server has no built-in TLS or
application authentication; use the [deployment guide](deployment.md) when
exposing it to other machines.

The Origin allowlist checks browser-origin requests. Loopback origins are
allowed; additional origins require explicit configuration. Origin checking
is not authentication for arbitrary clients.

Optional Prometheus metrics are served on a separate listener, defaulting to
`127.0.0.1:9090`. They expose operational counters/gauges/histograms, without
audio or transcript content. A configured scraper can transmit those metrics
to its own storage.

| Data | Built-in destination | Persistence |
|---|---|---|
| Audio | Engine on the processing host | Active buffers/jobs in memory; source file remains wherever the caller stored it |
| Transcript | CLI/binding caller or API client | Explicit output files; optional in-memory jobs |
| Logs | Configured process log output | Depends on terminal redirection, service journal or collector |
| Metrics | Configured metrics listener | Depends on the scraper |
| Models/caches | Local model and cache directories | Reused across runs |

See [sharing logs safely](troubleshooting.md#sharing-logs-safely) before
publishing diagnostics.
