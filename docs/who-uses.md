# Who uses gigastt

Public integrations found on September 15, 2026. This inventory covers
18 externally owned repositories with explicit gigastt code or deployment
configuration. The source links pin the inspected revisions.

Evidence ranges from operating reports to optional adapters and prototypes.
A repository entry does not establish an installation count or imply an
endorsement. Projects using GigaAM through other runtimes are outside this list.

## Assistants and dictation

| Project | Integration | Evidence |
|---|---|---|
| [janvarev/Irene-Voice-Assistant](https://github.com/janvarev/Irene-Voice-Assistant) | Russian voice assistant; Python WebSocket runner. | Merged integration; downstream author confirms configuration/use. [Source](https://github.com/janvarev/Irene-Voice-Assistant/blob/adb469ed045bfbc2de30b9b0bfa773cd2f7250bc/runva_gigastt.py) |
| [YFrtn/sotto](https://github.com/YFrtn/sotto) | macOS dictation; bundled server controlled from Swift, REST uploads. | Implemented adapter and process lifecycle. [Source](https://github.com/YFrtn/sotto/blob/ab6b92acb5960e7a04adb8ebbcd1f818e697ce37/sotto/Services/GigaSTTEngine.swift) |
| [monklock/deskpilot](https://github.com/monklock/deskpilot) | Windows voice control; C# with REST and WebSocket. | Provider replacement merged by the downstream author ([PR](https://github.com/monklock/deskpilot/pull/8)). [Source](https://github.com/monklock/deskpilot/blob/29e1f5d8eeae1550a4b5050811dbe11b438ce2d4/src/DeskPilot.Voice.GigaStt/GigaSttSpeechToTextProvider.cs) |
| [gumbertnntu-cloud/dictator](https://github.com/gumbertnntu-cloud/dictator) | macOS dictation; gigastt CLI as a fallback to gigaam-mlx. | Optional backend; active selection is unconfirmed. [Source](https://github.com/gumbertnntu-cloud/dictator/blob/8de62bf1096a0b643fe440e468238fad4c03a0e4/Sources/DictatorCore/Services/GigaAMRuntime.swift) |
| [smalever/typr](https://github.com/smalever/typr) | Linux dictation; Docker installation and an OpenAI-compatible client. | Optional backend in a fork with its own integration changes. [Source](https://github.com/smalever/typr/blob/7a2ae58d02a6122bc3f94e832354c169d71e18bc/install.sh) |
| [karamn0v/chrome-firefox-localstt](https://github.com/karamn0v/chrome-firefox-localstt) | Chrome/Firefox dictation through a server on a NAS. | Extension and Docker configuration; created September 15, 2026. [Source](https://github.com/karamn0v/chrome-firefox-localstt/blob/61d923676a39a48a98a7a73d8312e6024894a5f8/docker-compose.yml) |
| [klimlukichev/gigaam-voice-8765](https://github.com/klimlukichev/gigaam-voice-8765) | Browser recording interface with a Python service and Docker. | Recent prototype with an explicit gigastt path. [Source](https://github.com/klimlukichev/gigaam-voice-8765/blob/44aa800f7872807a9774cf123ea6819c3950fe7d/server.py) |
| [kekw2077/enhanced-voice-system](https://github.com/kekw2077/enhanced-voice-system) | Anima voice application; self-hosted OpenAI-compatible server. | Deployment script; [status notes](https://github.com/kekw2077/enhanced-voice-system/blob/7691829461813d0daf25389e8f356cbac74ad3d4/test1/server/stt/README.md) say the container was not tested locally. [Source](https://github.com/kekw2077/enhanced-voice-system/blob/7691829461813d0daf25389e8f356cbac74ad3d4/test1/server/stt/deploy-stt.sh) |

## Meetings, recordings and transcription workflows

| Project | Integration | Evidence |
|---|---|---|
| [levalovushka/propeller](https://github.com/levalovushka/propeller) | macOS meetings; two live audio sources, WebSocket partials and a final file pass. | Implemented integration and [reports of real meetings](https://github.com/levalovushka/propeller/blob/a08e3c8787ba770c8c77a889b4b57458830f3a4f/STATE.md). [Source](https://github.com/levalovushka/propeller/blob/a08e3c8787ba770c8c77a889b4b57458830f3a4f/meeting-recorder/swift/Sources/GigasttLiveSession.swift) |
| [luiz2047/OpenOffer](https://github.com/luiz2047/OpenOffer) | Electron interview/meeting workflow; a managed local server and WebSocket client. | Implemented client with process management and reconnection. [Source](https://github.com/luiz2047/OpenOffer/blob/34d505d56afee3caa03080a74b6bc26850d2597f/electron/audio/GigaSTTStreamingSTT.ts) |
| [biztrackru/diktum](https://github.com/biztrackru/diktum) | Interviews and recordings; Python CLI wrapper and external diarization. | Implemented adapter and pinned installer. [Source](https://github.com/biztrackru/diktum/blob/d763b1da7ace0de4414d37ec0d7fccf2c887334a/app/src/voice_recognizer/gigastt.py) |
| [voradori/GigaSTT-MediaTranscriber](https://github.com/voradori/GigaSTT-MediaTranscriber) | Windows file/video/YouTube transcription; CLI plus optional pyannote. | Implemented wrapper; created September 14, 2026. [Source](https://github.com/voradori/GigaSTT-MediaTranscriber/blob/b82a8a401626a9944f33425ce111204bec8e6d9b/scripts/transcribe.py) |
| [AndrewMoryakov/gigastt-pyannote-transcriber](https://github.com/AndrewMoryakov/gigastt-pyannote-transcriber) | Windows recordings with four speakers; CLI word timings plus pyannote. | Pinned executable adapter in a published project snapshot. [Source](https://github.com/AndrewMoryakov/gigastt-pyannote-transcriber/blob/f5dac913d49b5c08c0a57dd0c9c1b96edda80dfb/src/fourvoices/gigastt.py) |
| [voodoo2serg/recognition](https://github.com/voodoo2serg/recognition) | Phone recordings to Obsidian; Syncthing, a REST worker and local LLM processing. | Implemented worker, chunking and Docker configuration. [Source](https://github.com/voodoo2serg/recognition/blob/85a77e6744125d829f8175e259fc69fb0ca0334a/scripts/recognition-worker.sh) |
| [slapsh/podsum](https://github.com/slapsh/podsum) | Podcast transcription and summarization; Python UniFFI adapter. | Adapter exists; live gigastt inference is not confirmed. [Source](https://github.com/slapsh/podsum/blob/cadc1a10a47233c04da1e0d9077eea613bf0eded/podsum/asr/gigaam_backend.py) |

## Services and experiments

| Project | Integration | Evidence |
|---|---|---|
| [mazixs/stt-api](https://github.com/mazixs/stt-api) | Docker console/proxy around native and OpenAI-compatible endpoints. | [Operating notes](https://github.com/mazixs/stt-api/blob/c0cea08a577cc2dbf5fe746eb6c37af49842213d/docs/open-questions.md) and repeated measurements on real audio. [Source](https://github.com/mazixs/stt-api/blob/c0cea08a577cc2dbf5fe746eb6c37af49842213d/console/proxy.py) |
| [edinayasredarf-png/edinayasredanew2025](https://github.com/edinayasredarf-png/edinayasredanew2025) | CRM/call analysis; a selectable Python adapter to the OpenAI-compatible endpoint. | Adapter exists; its native speaker-label mode is unverified. [Source](https://github.com/edinayasredarf-png/edinayasredanew2025/blob/d8e08bb0c28b0968945ec21b36aea5600187b5d7/speech-service/app.py) |
| [kaifaty/NextEngine](https://github.com/kaifaty/NextEngine) | Speech timeline research; pinned gigastt server with a WebSocket adapter. | Optional comparison baseline, not the selected production game recognizer. [Source](https://github.com/kaifaty/NextEngine/blob/eb3287ff9cd5b07fcaea19d54fac2102099f056e/tools/speech-timeline/src/nextengine_speech_timeline/adapters/gigastt.py) |

## Documented compatibility and user reports

These are separate from the 18 repositories above.

| Project or workflow | What is established | Evidence |
|---|---|---|
| [VoiceStudio](https://github.com/debpalash/VoiceStudio) | Documents gigastt as a server for its generic OpenAI-compatible ASR adapter. This does not measure how many users select it. | [Integration guide](https://github.com/debpalash/VoiceStudio/blob/4e55180f700e2ce1b39195ec7377b9b3da8e20b2/docs/engines/openai-compatible-asr.md) |
| [OpenWhispr](https://github.com/OpenWhispr/openwhispr) | A Linux user reports running gigastt and testing WebM/WAV capture paths on real speech. The reporter also maintains stt-api above. | [User report](https://github.com/OpenWhispr/openwhispr/issues/2028#issuecomment-5542312111), [follow-up](https://github.com/OpenWhispr/openwhispr/issues/2028#issuecomment-5552847496) |
| Custom voice assistant | An operator describes a deployment behind a gRPC gateway on an A100. The report concerns older gigastt versions. | [Deployment and measurements](https://github.com/ekhodzitsky/gigastt/issues/307) |
| Telegram and long recordings | A user reports file transcription and confirms a long-file fix. | [Long-file report](https://github.com/ekhodzitsky/gigastt/issues/236) |
| OpenClaw trial | A user tested local Telegram transcription but explicitly stayed with the cloud for latency. This is an evaluated integration, not a completed migration. | [User's decision](https://github.com/ekhodzitsky/gigastt/issues/337#issuecomment-5661515847) |

## Scope and updates

Discovery combined GitHub repository, code, commit and issue searches for
`gigastt`, `gigastt-core` and `ekhodzitsky/gigastt`, followed by source review.
Private projects, maintainer-owned projects, copied repositories, catalogs,
proposal-only mentions and direct GigaAM integrations were excluded from
the 18-repository count. Optional implementations remain labeled as such.

Irene's initial integration was contributed by gigastt's maintainer and
accepted downstream: [runner](https://github.com/janvarev/Irene-Voice-Assistant/pull/75)
and [endpoint adjustment](https://github.com/janvarev/Irene-Voice-Assistant/pull/76).
Its author subsequently discussed [using the configuration](https://github.com/ekhodzitsky/gigastt/issues/202).

To add or correct an entry, open a pull request with a public repository
link and a source file, configuration or user report that demonstrates the
integration. Include whether it is deployed, optional or experimental.

[Back to README](../README.md)
