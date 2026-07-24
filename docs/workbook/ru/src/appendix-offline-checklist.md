# Приложение B — Чеклист offline / air-gapped

Когда хост не должен ходить в HuggingFace или GitHub в рантайме.
Подробные install-рецепты — в [Начало работы](01-getting-started.md) и
[Развёртывание](06-deployment-ops.md); здесь — операторский чеклист.

## На машине с сетью

- [ ] Тег релиза: `TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)` (или пин `v2.14.1`)
- [ ] Скачать **offline bundle** под arch  
      (`gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz` или aarch64)  
      **или** deb-пару: `gigastt_…_amd64.deb` + `gigastt-model-int8_…_all.deb`
- [ ] Скачать `.sha256` (+ `.minisig` при проверке minisign)
- [ ] Проверить checksums (`sha256sum -c`) и при желании
      [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md)
- [ ] Если нужна **диаризация**, отдельно заберите `wespeaker_resnet34.onnx`
      (не всегда в lean offline-бандле) и скопируйте в model dir —
      либо смиритесь с mono без `speaker`
- [ ] Если нужен **Silero VAD** offline — так же заранее заберите VAD-модель
      (`--vad` иначе полезет в сеть)

## На air-gapped хосте

- [ ] Установить бинарь + модель (installer, `dpkg -i`, или распаковка)
- [ ] Файлы в `~/.gigastt/models/` (или ваш `--model-dir`)
- [ ] Offline-гард: `GIGASTT_OFFLINE=1` / `--offline` — missing file
      **падает сразу**, а не висит на DNS
- [ ] Smoke: `gigastt transcribe sample.wav` (без сети)
- [ ] Сервер: `gigastt serve` → `curl http://127.0.0.1:9876/ready`
- [ ] systemd: unit из бандла; bind остаётся loopback, пока явно не
      `--bind-all` / `GIGASTT_ALLOW_BIND_ANY=1`

## Что остаётся offline в рантайме

| Действие | Сеть? |
|---|---|
| `transcribe` / batch / `watch` при моделях на диске | Нет |
| `serve` после наличия моделей | Нет |
| `POST /v1/admin/reload` после замены файлов | Нет |
| Первый `download` / нет punct / VAD / speaker | **Да** (режет `--offline`) |

## После копирования новых файлов модели

```sh
# Рестарт не обязателен, если serve уже поднят:
curl -s -X POST http://127.0.0.1:9876/v1/admin/reload
# {"reloaded":true,"variant":"rnnt","encoder":"int8"}
```

См. [Горячая перезагрузка](06-deployment-ops.md#горячая-перезагрузка-модели-без-рестарта).

## Ссылки

- [Начало работы — air-gapped](01-getting-started.md)
- [Развёртывание — offline](06-deployment-ops.md)
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md)
- [docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)
