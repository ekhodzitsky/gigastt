//! Pure pool / thread / RAM budgeting helpers for engine load.
//!
//! Free of ONNX sessions so unit tests can drive the math with plain inputs.
//! Called from [`super::engine::Engine`] load cascade.

use anyhow::Context;

/// Conservative RAM-cap budget: a multiple of the encoder file size on disk.
/// After mmap the measured extra slot is ~20 MB resident (not another encoder
/// copy); this multiplier stays pessimistic so [`cap_pool_size_for_ram`] still
/// clamps oversized `--pool-size` on small hosts. Used to keep
/// `pool_size * encoder_file_bytes * this` under a fraction of total RAM.
pub(crate) const ENCODER_RESIDENT_MULTIPLIER: u64 = 2;

/// Fraction (denominator) of total system RAM the pooled encoder sessions are
/// allowed to occupy before [`cap_pool_size_for_ram`] clamps the pool.
/// `2` = at most half of total RAM budgeted to encoder slots, leaving headroom
/// for the decoder/joiner sessions, audio buffers, inference arenas, and the
/// rest of the system.
pub(crate) const POOL_RAM_FRACTION_DENOM: u64 = 2;

/// Parse a cgroup memory limit file body (`memory.max` v2 or
/// `memory.limit_in_bytes` v1). Returns `None` for missing/unbounded/`max`.
/// Pure so unit tests can feed strings without a real cgroup mount.
#[cfg(any(test, target_os = "linux"))]
pub(crate) fn parse_cgroup_memory_limit(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("max") {
        return None;
    }
    let bytes: u64 = s.parse().ok()?;
    // Kernel v1 often reports a huge sentinel (~2^63-1) when unlimited.
    if bytes == 0 || bytes >= (1u64 << 62) {
        return None;
    }
    Some(bytes)
}

/// Read Linux cgroup memory limit (v2 then v1). `None` on non-Linux or when
/// unlimited / unreadable. Does not panic on missing files.
pub(crate) fn cgroup_memory_limit_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        const CANDIDATES: &[&str] = &[
            "/sys/fs/cgroup/memory.max",                   // cgroup v2 unified
            "/sys/fs/cgroup/memory/memory.limit_in_bytes", // cgroup v1
        ];
        for path in CANDIDATES {
            if let Ok(raw) = std::fs::read_to_string(path)
                && let Some(bytes) = parse_cgroup_memory_limit(&raw)
            {
                return Some(bytes);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Total physical RAM in bytes, or `0` if it can't be determined (in which case
/// the pool RAM cap is a no-op). macOS: `sysctl HW_MEMSIZE`; Linux/other unix:
/// `sysconf(_SC_PHYS_PAGES) * _SC_PAGESIZE`.
pub(crate) fn total_ram_bytes() -> u64 {
    // Miri does not implement `_SC_PHYS_PAGES`. Unknown RAM already disables
    // the pool clamp, which is the right answer inside the interpreter.
    #[cfg(miri)]
    {
        return 0;
    }
    #[cfg(not(miri))]
    {
        host_ram_bytes()
    }
}

#[cfg(not(miri))]
fn host_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut mem: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        // SAFETY: `mib`/`mem`/`len` are valid for the duration of the call;
        // sysctl writes at most `len` bytes into `mem`.
        let rc = unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut libc::c_int,
                mib.len() as libc::c_uint,
                &mut mem as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 { mem } else { 0 }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // SAFETY: sysconf has no side effects and returns -1 on error.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            (pages as u64).saturating_mul(page_size as u64)
        } else {
            0
        }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

/// Effective RAM budget for pool sizing: **min(host RAM, cgroup memory.max)**
/// when a cgroup limit is present (Docker/k8s). Falls back to host-only when
/// cgroup files are missing (macOS, bare metal without limits).
pub(crate) fn effective_ram_bytes() -> u64 {
    let host = total_ram_bytes();
    match cgroup_memory_limit_bytes() {
        Some(limit) if limit > 0 => {
            if host == 0 {
                limit
            } else {
                host.min(limit)
            }
        }
        _ => host,
    }
}

/// Number of triplets to reserve for the batch pool given `n` loaded and a
/// requested `batch_pool_size`, always leaving at least one for the
/// interactive pool (so `n <= 1` or `batch_pool_size == 0` yields 0).
pub(crate) fn batch_split_count(n: usize, batch_pool_size: usize) -> usize {
    batch_pool_size.min(n.saturating_sub(1))
}

/// Clamp the requested `pool_size` so the pooled encoder sessions can't
/// exceed [`POOL_RAM_FRACTION_DENOM`]⁻¹ of total RAM. Each triplet costs
/// about `encoder_bytes * ENCODER_RESIDENT_MULTIPLIER` resident (the encoder
/// dominates; decoder/joiner are small), so the max safe pool is
/// `(total_ram / denom) / per_triplet`, never below 1. Logs a warning when
/// it clamps. A no-op (returns `requested`) when RAM or encoder size is
/// unknown (`0`) — we never *raise* the requested size, only lower it.
pub(crate) fn cap_pool_size_for_ram(requested: usize, encoder_bytes: u64, total_ram: u64) -> usize {
    if requested <= 1 || encoder_bytes == 0 || total_ram == 0 {
        return requested.max(1);
    }
    let per_triplet = encoder_bytes.saturating_mul(ENCODER_RESIDENT_MULTIPLIER);
    let budget = total_ram / POOL_RAM_FRACTION_DENOM;
    // At least one slot always allowed even if a single triplet exceeds the
    // budget — the pool can't be empty, and partial-load tolerance
    // (`min_size`) handles a genuine OOM at load time.
    let max_slots = (budget / per_triplet.max(1)).max(1) as usize;
    if max_slots < requested {
        tracing::warn!(
            "Capping pool size {requested} -> {max_slots}: \
             {requested} encoder slots (~{} MiB each) would exceed half of \
             {} MiB total RAM. Concurrency is reduced; add RAM or lower \
             --pool-size to silence this.",
            per_triplet / (1024 * 1024),
            total_ram / (1024 * 1024),
        );
        max_slots
    } else {
        requested
    }
}

/// Clamp the requested encoder intra-op thread count so the pooled encoder
/// sessions can't oversubscribe the CPU. Each of `pool_size` triplets can
/// run concurrently, so the total intra-op parallelism is
/// `pool_size * threads`; capping that at `logical_cpus` keeps the machine
/// from thrashing on context switches. The effective per-encoder count is
/// therefore `clamp(requested, 1, logical_cpus / pool_size)`, never below 1.
/// Logs a warning when it lowers the request. The default `requested == 1`
/// always returns `1`, so the built sessions are unchanged.
pub(crate) fn clamp_encoder_intra_threads(
    pool_size: usize,
    requested: usize,
    logical_cpus: usize,
) -> usize {
    let requested = requested.max(1);
    let pool_size = pool_size.max(1);
    let logical_cpus = logical_cpus.max(1);
    // Leave at least one thread per encoder even on a machine with fewer
    // logical CPUs than pooled triplets.
    let max_per_encoder = (logical_cpus / pool_size).max(1);
    if requested > max_per_encoder {
        tracing::warn!(
            "Capping encoder intra-op threads {requested} -> {max_per_encoder}: \
             {pool_size} pooled encoder(s) x {requested} threads would exceed \
             the {logical_cpus} logical CPU(s) available. Lower --pool-size or \
             --encoder-intra-threads to silence this."
        );
        max_per_encoder
    } else {
        requested
    }
}

/// Partition `items` into an interactive pool and an optional batch pool of
/// `batch_pool_size` items, always leaving at least one item interactive.
/// `batch_pool_size == 0` (or too few items to split) yields no batch pool.
pub(crate) fn split_pool_items<T: Send>(
    mut items: Vec<T>,
    batch_pool_size: usize,
) -> (Vec<T>, Option<Vec<T>>) {
    let n = items.len();
    let batch = batch_split_count(n, batch_pool_size);
    if batch == 0 {
        return (items, None);
    }
    let batch_items = items.split_off(n - batch);
    (items, Some(batch_items))
}

/// Probe a freshly-built state; on failure, rebuild it once and re-probe.
///
/// `probe` is a runtime self-check, `rebuild` converts the failed state into
/// a replacement (receiving the probe error so it can log the cause). A
/// rebuilt state that still fails the probe is a hard error — there is no
/// second fallback level.
///
/// Production call site is CoreML runtime-fallback; unit tests exercise the
/// pure decision logic without a session.
#[cfg_attr(not(any(test, feature = "coreml")), allow(dead_code))]
pub(crate) fn probe_or_rebuild<S>(
    state: S,
    probe: impl Fn(&S) -> anyhow::Result<()>,
    rebuild: impl FnOnce(S, anyhow::Error) -> anyhow::Result<S>,
) -> anyhow::Result<S> {
    match probe(&state) {
        Ok(()) => Ok(state),
        Err(probe_err) => {
            let rebuilt = rebuild(state, probe_err)?;
            probe(&rebuilt).context("state failed probe even after rebuild")?;
            Ok(rebuilt)
        }
    }
}

/// Decide the final pool from per-triplet load results: returns the
/// successfully loaded triplets when at least `min_size` loaded (warning
/// when the pool is degraded below `pool_size`), or an error describing the
/// shortfall. `min_size` is clamped to `1..=pool_size`.
pub(crate) fn finalize_pool_load<T>(
    results: Vec<anyhow::Result<T>>,
    pool_size: usize,
    min_size: usize,
) -> anyhow::Result<Vec<T>> {
    let min_size = min_size.clamp(1, pool_size.max(1));
    let mut loaded = Vec::with_capacity(results.len());
    let mut first_err: Option<anyhow::Error> = None;
    for r in results {
        match r {
            Ok(t) => loaded.push(t),
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }
    let n = loaded.len();
    if n >= min_size {
        if n < pool_size {
            let detail = first_err
                .map(|e| format!("; first error: {e:#}"))
                .unwrap_or_default();
            tracing::warn!(
                "degraded pool: loaded {n}/{pool_size} session triplets ({} failed){detail}",
                pool_size - n
            );
        }
        Ok(loaded)
    } else {
        let detail = first_err.map(|e| format!(": {e:#}")).unwrap_or_default();
        Err(anyhow::anyhow!(
            "loaded only {n}/{pool_size} session triplets, need at least {min_size}{detail}"
        ))
    }
}

#[cfg(test)]
mod tests;
