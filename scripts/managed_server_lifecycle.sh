#!/usr/bin/env bash
# App-owned gigastt process: loopback serve, INT8 model, writable ORT cache.
#
# Phases (one child at a time; this process reaps every child):
#   1. startup failure — empty model dir + offline, error names the INT8 path
#   2. successful POST /v1/transcribe
#   3. restart with GIGASTT_OFFLINE=1 and transcribe again
#   4. SIGKILL during early startup; wait(2) reaps the child
#   5. SIGTERM after /ready; no owned gigastt child remains
#
# Does not download models and does not delete anything under the model dir.
# Port is fixed at 9881. See docs/managed-lifecycle.md.

set -euo pipefail

PORT=9881
HOST=127.0.0.1
BIN=${GIGASTT_BIN:-gigastt}
MODEL_DIR=${GIGASTT_MODEL_DIR:-${HOME}/.gigastt/models}
READY_TIMEOUT=${GIGASTT_READY_TIMEOUT_SECS:-240}

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
AUDIO=${GIGASTT_AUDIO:-${ROOT}/crates/gigastt/tests/fixtures/golos_00.wav}

server_pid=""
server_log=""
work=""

die() {
    printf 'lifecycle: %s\n' "$*" >&2
    if [[ -n "${server_log:-}" && -f "$server_log" ]]; then
        printf '%s\n' '--- server log (tail) ---' >&2
        tail -n 80 "$server_log" >&2 || true
    fi
    exit 1
}

cpu_model() {
    if [[ -r /proc/cpuinfo ]]; then
        awk -F: '/model name/ { gsub(/^[ \t]+/, "", $2); print $2; exit }' /proc/cpuinfo
    else
        uname -m
    fi
}

port_hex() {
    printf '%04X' "$PORT"
}

# Print local addresses of LISTEN sockets on $PORT (state 0A).
port_listeners() {
    local hex
    hex=$(port_hex)
    awk -v p="$hex" '
        NR > 1 && $4 == "0A" {
            split($2, a, ":")
            if (toupper(a[2]) == p) print toupper(a[1])
        }
    ' /proc/net/tcp /proc/net/tcp6 2>/dev/null || true
}

assert_port_free() {
    local listeners
    listeners=$(port_listeners)
    if [[ -n "$listeners" ]]; then
        die "port ${PORT} is already in use (${listeners}); not starting a server"
    fi
}

assert_loopback_only() {
    local listeners addr
    listeners=$(port_listeners)
    [[ -n "$listeners" ]] || die "expected a listener on port ${PORT}"
    while IFS= read -r addr; do
        [[ -n "$addr" ]] || continue
        # 127.0.0.1 is little-endian 0100007F in /proc/net/tcp.
        if [[ "$addr" != "0100007F" ]]; then
            die "port ${PORT} is not loopback-only (local address ${addr})"
        fi
    done <<<"$listeners"
}

stop_server() {
    local pid="${server_pid:-}"
    local i st
    [[ -n "$pid" ]] || return 0
    server_pid=""
    if kill -0 "$pid" 2>/dev/null; then
        st=$(ps -o stat= -p "$pid" 2>/dev/null | awk '{ print $1 }' || true)
        if [[ "$st" != Z* ]]; then
            kill -TERM "$pid" 2>/dev/null || true
            for ((i = 0; i < 80; i++)); do
                st=$(ps -o stat= -p "$pid" 2>/dev/null | awk '{ print $1 }' || true)
                [[ -z "$st" || "$st" == Z* ]] && break
                sleep 0.25
            done
            st=$(ps -o stat= -p "$pid" 2>/dev/null | awk '{ print $1 }' || true)
            if [[ -n "$st" && "$st" != Z* ]]; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
    fi
    wait "$pid" 2>/dev/null || true
}

on_err() {
    printf 'lifecycle: failed at line %s\n' "$1" >&2
    if [[ -n "${server_log:-}" && -f "$server_log" ]]; then
        printf '%s\n' '--- server log (tail) ---' >&2
        tail -n 60 "$server_log" >&2 || true
    fi
}

cleanup() {
    stop_server
    if [[ -n "${work:-}" && -d "$work" ]]; then
        rm -rf "$work"
    fi
}

trap 'on_err $LINENO' ERR
trap cleanup EXIT

require_tools() {
    command -v "$BIN" >/dev/null 2>&1 || die "gigastt binary not found: ${BIN}"
    command -v curl >/dev/null 2>&1 || die "curl is required"
    command -v python3 >/dev/null 2>&1 || die "python3 is required"
    [[ -r /proc/net/tcp ]] || die "this script reads /proc/net/tcp (Linux only)"
    [[ -f "$AUDIO" ]] || die "audio fixture not found: ${AUDIO}"
}

require_int8_rnnt() {
    local name path
    [[ -d "$MODEL_DIR" ]] || die "model dir not found: ${MODEL_DIR} (run: gigastt download --model-dir <dir>)"
    for name in \
        v3_rnnt_encoder_int8.onnx \
        v3_rnnt_decoder.onnx \
        v3_rnnt_joint.onnx \
        v3_vocab.txt
    do
        path="${MODEL_DIR}/${name}"
        [[ -f "$path" ]] || die "missing INT8 rnnt file ${path} (run: gigastt download --model-dir ${MODEL_DIR})"
    done
    # The cache must be a different writable directory. Never point it at the
    # model tree: a read-only model install cannot host the ORT graph cache.
    case "$cache" in
        "$MODEL_DIR" | "$MODEL_DIR"/*)
            die "optimized cache dir must not be inside the model dir"
            ;;
    esac
}

reject_removed_download_flags() {
    local flag out rc empty
    empty="${work}/flag-probe"
    mkdir -p "$empty"
    for flag in --fp32 --prequantized --skip-quantize; do
        out="${work}/flag${flag}.txt"
        set +e
        "$BIN" download "$flag" --model-dir "$empty" >"$out" 2>&1
        rc=$?
        set -e
        if [[ "$rc" -eq 0 ]]; then
            die "download ${flag} was accepted; refusing to continue"
        fi
        if ! grep -q "unexpected argument '${flag}'" "$out"; then
            die "download ${flag} was not rejected as an unexpected argument (exit ${rc})"
        fi
        if find "$empty" -type f | grep -q .; then
            die "download ${flag} wrote into ${empty}"
        fi
        printf 'RESULT retired_flag flag=%s exit=%s\n' "$flag" "$rc"
    done
}

serve_common=(
    --host "$HOST"
    --port "$PORT"
    --pool-size 1
    --model-variant rnnt
    --punctuation off
    --itn off
    --log-level info
)

start_server() {
    local offline="$1"
    local model_dir="$2"
    local label="$3"
    assert_port_free
    server_log="${work}/${label}.log"
    : >"$server_log"
    if [[ "$offline" == 1 ]]; then
        env GIGASTT_OFFLINE=1 "$BIN" serve \
            "${serve_common[@]}" \
            --model-dir "$model_dir" \
            --optimized-cache-dir "$cache" \
            >"$server_log" 2>&1 &
    else
        env -u GIGASTT_OFFLINE "$BIN" serve \
            "${serve_common[@]}" \
            --model-dir "$model_dir" \
            --optimized-cache-dir "$cache" \
            >"$server_log" 2>&1 &
    fi
    server_pid=$!
    printf 'started pid=%s offline=%s model_dir=%s log=%s\n' \
        "$server_pid" "$offline" "$model_dir" "$server_log"
}

refuse_if_downloading() {
    if [[ -f "$server_log" ]] && grep -E -q 'Downloading |downloading pre-quantized|Downloading pre-quantized' "$server_log"; then
        die "server started a model download; this recipe does not fetch weights"
    fi
}

http_code() {
    local path="$1"
    local body="$2"
    local code
    # -s (not -sS): probes run before the socket is bound and must not spam.
    code=$(curl -s -m 2 -o "$body" -w '%{http_code}' "http://${HOST}:${PORT}${path}" || true)
    printf '%s' "${code:-000}"
}

# True when pid is gone or a zombie waiting to be reaped by this shell.
pid_exited() {
    local st
    st=$(ps -o stat= -p "$1" 2>/dev/null | awk '{ print $1 }' || true)
    [[ -z "$st" || "$st" == Z* ]]
}

wait_ready() {
    local pid="$1"
    local deadline=$((SECONDS + READY_TIMEOUT))
    local health_code ready_code saw_init=0
    health_code=000
    ready_code=000
    while ((SECONDS < deadline)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            die "server pid ${pid} exited before /ready"
        fi
        refuse_if_downloading
        health_code=$(http_code /health "${work}/health.json")
        ready_code=$(http_code /ready "${work}/ready.json")
        if [[ "$ready_code" == 503 ]]; then
            saw_init=1
        fi
        if [[ "$health_code" == 200 && "$ready_code" == 200 ]]; then
            python3 - "$work/health.json" "$work/ready.json" <<'PY'
import json, sys
health = json.load(open(sys.argv[1]))
ready = json.load(open(sys.argv[2]))
if health.get("status") != "ok" or health.get("variant") != "rnnt":
    raise SystemExit(f"unexpected /health: {health}")
if health.get("model") != "gigaam-v3-rnnt":
    raise SystemExit(f"unexpected model id: {health}")
if ready.get("status") != "ready" or int(ready.get("pool_available") or 0) < 1:
    raise SystemExit(f"unexpected /ready: {ready}")
PY
            printf 'RESULT ready health=%s ready=%s initializing_seen=%s variant=rnnt\n' \
                "$health_code" "$ready_code" "$saw_init"
            return 0
        fi
        sleep 0.3
    done
    die "timed out waiting for /ready (last health=${health_code} ready=${ready_code})"
}

transcribe_once() {
    local label="$1"
    local body="${work}/${label}.json"
    local code
    code=$(curl -sS -m 90 -o "$body" -w '%{http_code}' \
        -X POST \
        -H 'Content-Type: application/octet-stream' \
        --data-binary @"$AUDIO" \
        "http://${HOST}:${PORT}/v1/transcribe" || true)
    if [[ "$code" != 200 ]]; then
        printf 'transcribe body:\n' >&2
        cat "$body" >&2 || true
        die "${label} transcribe HTTP ${code}"
    fi
    python3 - "$body" "$label" <<'PY'
import json, sys
path, label = sys.argv[1], sys.argv[2]
doc = json.load(open(path))
text = doc.get("text")
if not isinstance(text, str) or not text.strip():
    raise SystemExit(f"{label}: response has no non-empty text")
print(f"RESULT {label} http=200 text_chars={len(text)}")
PY
}

assert_cache_writable() {
    local probe="${cache}/.write-probe-parent"
    : >"$probe"
    rm -f "$probe"
    if grep -E -q 'optimized graph cache directory (cannot be created|is not writable)|optimized graph cache write failed' "$server_log"; then
        die "server degraded the ORT cache; expected a writable cache dir"
    fi
    local graphs
    graphs=$(find "$cache" -maxdepth 1 -type f -name '*_optimized.ort' -printf '%f\n' || true)
    [[ -n "$graphs" ]] || die "writable cache dir has no *_optimized.ort after /ready"
    printf 'RESULT cache dir=%s files=%s\n' "$cache" "$(echo "$graphs" | tr '\n' ' ')"
}

assert_no_owned_gigastt() {
    local kids
    kids=$(ps -o pid=,ppid=,stat=,cmd= -u "$(id -un)" | awk -v p="$$" '$2 == p && /gigastt/ { print }' || true)
    if [[ -n "$kids" ]]; then
        printf '%s\n' "$kids" >&2
        die "owned gigastt child still running"
    fi
    printf 'RESULT children none\n'
}

phase_startup_failure() {
    local empty pid rc health_code ready_code err_line saw_health=0 saw_ready=0 i
    empty="${work}/empty-model"
    mkdir -p "$empty"
    start_server 1 "$empty" startup-failure
    pid=$server_pid
    health_code=000
    ready_code=000
    for ((i = 0; i < 80; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        health_code=$(http_code /health "${work}/fail-health.json")
        ready_code=$(http_code /ready "${work}/fail-ready.json")
        [[ "$health_code" == 200 ]] && saw_health=1
        [[ "$ready_code" == 503 ]] && saw_ready=1
        sleep 0.05
    done
    if ! pid_exited "$pid"; then
        kill -KILL "$pid" 2>/dev/null || true
        server_pid=""
        wait "$pid" 2>/dev/null || true
        die "startup failure did not exit on its own"
    fi
    server_pid=""
    rc=0
    wait "$pid" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        die "empty model dir exited 0; expected a startup failure"
    fi
    err_line=$(grep -F -m 1 "${empty}/v3_rnnt_encoder_int8.onnx" "$server_log" || true)
    if [[ -z "$err_line" ]]; then
        die "startup failure did not name ${empty}/v3_rnnt_encoder_int8.onnx"
    fi
    # Offline refusal takes the download lock before it bails, so an empty
    # .download.lock may remain. Weights and .partial files must not.
    local unexpected
    unexpected=$(find "$empty" -type f ! -name '.download.lock' -print || true)
    if [[ -n "$unexpected" ]]; then
        die "startup failure wrote model files: ${unexpected}"
    fi
    if ps -p "$pid" >/dev/null 2>&1; then
        die "startup-failure pid ${pid} was not reaped"
    fi
    printf 'RESULT startup_failure exit=%s health_200=%s ready_503=%s\n' \
        "$rc" "$saw_health" "$saw_ready"
    printf 'RESULT startup_failure_line %s\n' "$err_line"
}

phase_success() {
    start_server 0 "$MODEL_DIR" success
    wait_ready "$server_pid"
    assert_loopback_only
    assert_cache_writable
    transcribe_once success
    refuse_if_downloading
}

phase_offline_restart() {
    local previous=$server_pid
    stop_server
    if ps -p "$previous" >/dev/null 2>&1; then
        die "pre-offline pid ${previous} was not reaped"
    fi
    assert_port_free
    start_server 1 "$MODEL_DIR" offline
    wait_ready "$server_pid"
    assert_loopback_only
    transcribe_once offline_restart
    refuse_if_downloading
    if ! grep -q 'GIGASTT_OFFLINE' <<<"$(tr '\0' '\n' <"/proc/${server_pid}/environ" 2>/dev/null || true)"; then
        # environ is the child's; the flag is set inside the process after
        # --offline, but this phase passes the env var directly.
        die "offline child is missing GIGASTT_OFFLINE in its environment"
    fi
}

phase_early_exit() {
    local pid rc st_before st_after i
    stop_server
    assert_port_free
    start_server 1 "$MODEL_DIR" early
    pid=$server_pid
    for ((i = 0; i < 50; i++)); do
        kill -0 "$pid" 2>/dev/null && break
        sleep 0.02
    done
    kill -0 "$pid" 2>/dev/null || die "early-exit server was not alive"
    kill -KILL "$pid" 2>/dev/null || true
    st_before=""
    for ((i = 0; i < 30; i++)); do
        st_before=$(ps -o stat= -p "$pid" 2>/dev/null | awk '{ print $1 }' || true)
        [[ "$st_before" == Z* ]] && break
        [[ -z "$st_before" ]] && break
        sleep 0.05
    done
    server_pid=""
    rc=0
    wait "$pid" || rc=$?
    st_after=$(ps -o stat= -p "$pid" 2>/dev/null | awk '{ print $1 }' || true)
    if ps -p "$pid" >/dev/null 2>&1; then
        die "early-exit pid ${pid} still present after wait (stat ${st_after:-unknown})"
    fi
    # 128+9 = 137 when the kernel reports SIGKILL. Accept any non-zero reap.
    if [[ "$rc" -eq 0 ]]; then
        die "early-exit wait status was 0"
    fi
    printf 'RESULT early_exit wait_status=%s stat_before_wait=%s stat_after_wait=%s\n' \
        "$rc" "${st_before:-none}" "${st_after:-gone}"
    assert_port_free
}

phase_clean_shutdown() {
    local pid rc
    assert_port_free
    start_server 1 "$MODEL_DIR" shutdown
    wait_ready "$server_pid"
    pid=$server_pid
    kill -TERM "$pid" 2>/dev/null || true
    server_pid=""
    rc=0
    # Reap only. stop_server would send a second signal. A zombie still
    # answers kill -0, so the exit check is "gone or Z", then wait.
    for ((i = 0; i < 80; i++)); do
        if pid_exited "$pid"; then
            break
        fi
        sleep 0.25
    done
    if ! pid_exited "$pid"; then
        die "clean shutdown did not exit after SIGTERM"
    fi
    wait "$pid" || rc=$?
    if ps -p "$pid" >/dev/null 2>&1; then
        die "clean-shutdown pid ${pid} was not reaped"
    fi
    assert_port_free
    printf 'RESULT clean_shutdown wait_status=%s\n' "$rc"
}

main() {
    require_tools
    work=$(mktemp -d "${TMPDIR:-/tmp}/gigastt-lifecycle.XXXXXX")
    cache="${work}/cache"
    mkdir -p "$cache"
    require_int8_rnnt
    assert_port_free
    # Fail the run if serve writes into the model dir (cache must stay outside).
    find "$MODEL_DIR" -printf '%T@ %s %p\n' | sort >"${work}/model.snap"

    printf 'RESULT platform %s\n' "$(uname -srm)"
    printf 'RESULT cpu %s\n' "$(cpu_model)"
    bin_path=$(command -v "$BIN")
    printf 'RESULT binary %s\n' "$bin_path"
    printf 'RESULT version %s\n' "$("$BIN" --version)"
    printf 'RESULT binary_sha256 %s\n' "$(sha256sum "$bin_path" | awk '{ print $1 }')"
    printf 'RESULT model_dir %s\n' "$MODEL_DIR"
    printf 'RESULT audio %s\n' "$AUDIO"
    printf 'RESULT port %s pool_size 1 host %s\n' "$PORT" "$HOST"

    reject_removed_download_flags
    phase_startup_failure
    phase_success
    phase_offline_restart
    phase_early_exit
    phase_clean_shutdown
    assert_no_owned_gigastt
    find "$MODEL_DIR" -printf '%T@ %s %p\n' | sort >"${work}/model.snap.after"
    if ! cmp -s "${work}/model.snap" "${work}/model.snap.after"; then
        diff -u "${work}/model.snap" "${work}/model.snap.after" >&2 || true
        die "model directory changed; this recipe must not write weights or caches there"
    fi
    printf 'RESULT model_dir_unchanged\n'
    printf 'RESULT ok\n'
}

main "$@"
