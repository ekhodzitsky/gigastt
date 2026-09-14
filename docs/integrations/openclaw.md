# OpenClaw: local voice transcription

Use OpenClaw's `openai` audio provider with gigastt's
`/v1/audio/transcriptions` endpoint. Select a separate audio credential profile
and put the local URL on the audio entry. The agent can keep its existing
ChatGPT/Codex OAuth profile and chat destination.

This recipe targets **OpenClaw v2026.8.2** and **gigastt 2.21.0**. OpenClaw's
provider plugins and network policy vary by version; check the commands below
on the gateway host after upgrading.

## Start gigastt and check the endpoint

Install gigastt using the [quick start](../../README.md#quickstart), then run:

```sh
gigastt serve --punctuation on --itn on
```

Wait for `GET /ready` to return HTTP 200. `/health` is also available while the
model is loading, so it does not establish inference readiness.

Run this from the OpenClaw gateway's network environment, replacing `voice.ogg`
with a Russian voice note:

```sh
curl --fail-with-body http://127.0.0.1:9876/ready
curl --fail-with-body http://127.0.0.1:9876/v1/audio/transcriptions \
  -H 'Authorization: Bearer sk-local-gigastt' \
  -F 'file=@voice.ogg' \
  -F 'model=whisper-1' \
  -F 'language=ru'
```

Expect `{"text":"..."}` with a nonempty transcript. gigastt accepts Telegram
OGG/Opus files directly. It does not require a key: `sk-local-gigastt` is a
placeholder for OpenClaw's credential selection. Its `sk-` prefix satisfies
the OpenAI key-format check in `models auth paste-api-key`; it is not a real
cloud credential. `whisper-1` is also a compatibility label; gigastt uses the
model already loaded by the server.

## Keep chat OAuth and select a separate audio profile

The example agent is `main`. Repeat the profile setup for each agent that
receives audio:

```sh
openclaw models auth list --agent main --provider openai
openclaw models auth paste-api-key --agent main --provider openai \
  --profile-id openai:gigastt
```

Enter `sk-local-gigastt` at the credential prompt. From the first command, copy the
exact existing OAuth profile ID, then keep it first in the auth order. Replace
`EXISTING_OAUTH_PROFILE_ID` below; retain any other profiles you already use:

```sh
openclaw models auth order set --agent main --provider openai \
  EXISTING_OAUTH_PROFILE_ID openai:gigastt
```

Merge this fragment into the existing OpenClaw configuration. Preserve unrelated
settings and any image/video entries in `tools.media.models`:

```json
{
  "models": {
    "providers": {
      "openai": {
        "request": { "allowPrivateNetwork": true }
      }
    }
  },
  "tools": {
    "media": {
      "audio": { "enabled": true },
      "models": [
        {
          "provider": "openai",
          "model": "whisper-1",
          "profile": "openai:gigastt",
          "baseUrl": "http://127.0.0.1:9876/v1",
          "capabilities": ["audio"],
          "language": "ru",
          "timeoutSeconds": 60
        }
      ]
    }
  }
}
```

The adapter appends `/audio/transcriptions`, so `baseUrl` must end at `/v1`.
Leave the provider-wide chat `baseUrl` and credentials configured for chat.
Remove any experimental provider-wide placeholder `apiKey` override:
`models.providers.openai.apiKey` takes precedence over the audio profile.

In v2026.8.2, the batch audio request path needs the explicit
`models.providers.openai.request.allowPrivateNetwork` setting to reach loopback.
Its scope is **all requests using the `openai` provider**, although the fragment
changes only the audio destination. The flag is accepted on the model-provider
request object; placing it under the audio entry's `request` or
`tools.media.audio.request` fails configuration validation.

OpenClaw documents the separate-profile setup in its
[versioned audio guide](https://github.com/openclaw/openclaw/blob/v2026.8.2/docs/nodes/audio.md#openai-transcription-alongside-chatgptcodex-oauth).
The URL and credential precedence follow the
[media entry runner](https://github.com/openclaw/openclaw/blob/v2026.8.2/src/media-understanding/runner.entries.ts).
The network setting is defined by the
[configuration schema](https://github.com/openclaw/openclaw/blob/v2026.8.2/src/config/zod-schema.core.ts)
and applied by the
[request policy](https://github.com/openclaw/openclaw/blob/v2026.8.2/src/agents/provider-request-config.ts)
and [audio HTTP helper](https://github.com/openclaw/openclaw/blob/v2026.8.2/src/media-understanding/shared.ts).

## Verify OpenClaw's route

Ensure the `openai` plugin is enabled and permitted by any plugin allowlist.
Validate and reload the gateway configuration using the commands for your
installation, then run:

```sh
openclaw capability audio providers --agent main --json
openclaw capability audio transcribe --agent main --file voice.ogg --json
```

Confirm that the provider inventory includes OpenAI audio support and that the
transcript comes from gigastt. Send a Telegram voice note, then an ordinary text
message: transcription should preserve gigastt punctuation/ITN, and chat should
continue using the existing OAuth route.

For local-only STT, keep gigastt as the only eligible audio entry and remove
cloud audio fallbacks. Test once with gigastt stopped: transcription should fail
instead of choosing a cloud provider. The transcript is still supplied to the
agent's chat model; local STT alone does not make the rest of OpenClaw local.

## Containers and proxies

`127.0.0.1` refers to the gateway's own network namespace. If gigastt runs in
another container, use its service hostname on a shared private network. If it
runs on the host, use an address reachable from the gateway container. Test
`/ready` and the multipart request from that same environment.

For a non-loopback listener, gigastt requires explicit opt-in, for example
`gigastt serve --bind-all --host 0.0.0.0`. Restrict access through the container
network or host firewall; the endpoint has no API-key authentication. The
`sk-local-gigastt` placeholder does not protect it.

If the gateway uses an HTTP proxy, arrange a direct route to the local STT
destination. Check its `NO_PROXY`/`no_proxy` settings and any explicit provider
proxy configuration; a curl test from a different shell may use a different
proxy policy.

## Troubleshooting

| Symptom | Check |
|---|---|
| `Media provider not available` for `gigastt` | A `models.providers.gigastt` entry configures a provider but does not register an audio implementation. Use the bundled `openai` audio provider. An HTTP shim cannot add a client-side capability. |
| `openai` missing from the audio inventory | Check the OpenAI plugin installation, enablement, and plugin allowlist. A working chat login alone does not verify its audio capability. |
| `groq` missing from the audio inventory | That version distributes Groq separately. A configuration entry alone does not install its provider plugin; this recipe does not require Groq. |
| OAuth/API-key error | Check the exact agent and `profile` ID. Use the dedicated placeholder profile and check for provider-wide credential overrides. |
| Private/loopback address rejected | Check the provider-level `request.allowPrivateNetwork` flag and validate its placement. |
| HTTP 404 | Use `http://HOST:9876/v1` as the base URL; avoid a duplicated `/v1` or a full transcription endpoint as the base. |
| Connection refused or readiness unavailable | Check service startup, port, container addressing, and proxy routing from the gateway environment. |
| CLI wrapper and `/bin/echo` both fail with `EACCES` | Diagnose process execution under the gateway's actual service user: directory permissions, executable permissions, mount options, and service/container policy. This is separate from HTTP transcription and does not establish a gigastt protocol defect. |

## Validation scope

The local HTTP smoke check used the repository's `speech_telegram.ogg` fixture
(31,758 bytes), an INT8 RNN-T model, punctuation on, and ITN on. Both
`model=whisper-1` with the placeholder bearer header and `model=gigastt` without
authentication returned HTTP 200 and the same text as `/v1/transcribe`:
`60000 тенге, сколько будет стоить?`.

The OpenClaw CLI check used the published `openclaw@2026.8.2` package with
Node.js 22.22.3 and isolated configuration/auth state. Configuration validation,
saving the `sk-local-gigastt` audio profile, and the audio provider inventory
succeeded. `capability audio transcribe` returned the same transcript through
gigastt. Setting `allowPrivateNetwork: false` produced `SsrFBlockedError`;
stopping gigastt produced a fetch error with this sole configured audio entry.

Telegram delivery and an existing OAuth chat account still need verification
in the deployment where they are configured.
