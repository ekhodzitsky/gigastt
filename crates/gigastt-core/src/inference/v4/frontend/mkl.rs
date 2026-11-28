//! Narrow GNU/Linux loader and FFI seam for one exact side-loaded DFTI group.
//!
//! The group is trusted executable code, not an untrusted model pack. The
//! operator must keep the directory immutable for the process lifetime. No
//! `dlclose`/hot-reload is promised; the qualified vendor group retained maps
//! after a successful close, so we intentionally hold one process-lifetime
//! namespace and never attempt to replace it.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::ffi::{CStr, CString, OsStr, c_char, c_void};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use super::AlignedFloats;
use crate::sha256::{Sha256, hex_lower};

const PROFILE: &str = include_str!("mkl-profile.json");
const SYSTEM_NAMES: &[&str] = &[
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "libpthread.so.0",
    "ld-linux-x86-64.so.2",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    file: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    schema_version: u32,
    #[serde(rename = "wheel")]
    _wheel: Pin,
    version: String,
    shim_abi: u32,
    libraries: Vec<Pin>,
    shim: Pin,
    message_catalog: Pin,
}

pub(super) struct VerifiedRuntime {
    dir: PathBuf,
    profile: Profile,
    expected_mapped: BTreeSet<PathBuf>,
}

fn regular_dir(path: &Path) -> Result<()> {
    ensure!(
        std::fs::symlink_metadata(path)?.is_dir(),
        "runtime directory must not be a symlink"
    );
    Ok(())
}

fn verify_file(dir: &Path, pin: &Pin) -> Result<()> {
    let path = dir.join(&pin.file);
    let meta = std::fs::symlink_metadata(&path)
        .with_context(|| format!("missing MKL artifact {}", pin.file))?;
    ensure!(
        meta.is_file() && meta.len() == pin.bytes,
        "invalid MKL artifact size or type: {}",
        pin.file
    );
    let file = std::fs::File::open(&path)?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.len() == pin.bytes,
        "MKL artifact changed before reading: {}",
        pin.file
    );
    let limit = pin
        .bytes
        .checked_add(1)
        .context("MKL artifact size overflow")?;
    let mut reader = file.take(limit);
    let mut hash = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut count = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        count += n as u64;
        hash.update(&buf[..n]);
    }
    ensure!(
        count == pin.bytes && hex_lower(&hash.finalize()) == pin.sha256,
        "MKL artifact SHA-256 mismatch: {}",
        pin.file
    );
    Ok(())
}

pub(super) fn validate_runtime_dir(dir: &Path) -> Result<VerifiedRuntime> {
    regular_dir(dir).context("invalid MKL runtime directory")?;
    let dir = dir.canonicalize()?;
    let profile: Profile = serde_json::from_str(PROFILE).context("invalid built-in MKL profile")?;
    ensure!(
        profile.schema_version == 1 && profile.libraries.len() == 12 && profile.shim_abi == 1,
        "invalid built-in MKL profile"
    );
    let allowed: BTreeSet<_> = profile
        .libraries
        .iter()
        .chain(std::iter::once(&profile.shim))
        .map(|p| OsStr::new(&p.file))
        .chain([OsStr::new("locale"), OsStr::new("license")])
        .collect();
    // $ORIGIN search includes glibc-hwcaps overlays. Reject unexpected root
    // entries before loading any code, not merely after a constructor runs.
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        ensure!(
            allowed.contains(entry.file_name().as_os_str()),
            "unexpected MKL runtime directory entry"
        );
        let ty = entry.file_type()?;
        ensure!(
            !ty.is_symlink(),
            "symlinks are not supported in MKL runtime directory"
        );
        if entry.file_name() == "locale" || entry.file_name() == "license" {
            ensure!(ty.is_dir(), "invalid runtime directory entry type");
        } else {
            ensure!(ty.is_file(), "invalid runtime artifact type");
        }
    }
    regular_dir(&dir.join("locale"))?;
    regular_dir(&dir.join("locale/en_US"))?;
    let mut expected_mapped = BTreeSet::new();
    for pin in profile
        .libraries
        .iter()
        .chain(std::iter::once(&profile.shim))
    {
        verify_file(&dir, pin)?;
        expected_mapped.insert(dir.join(&pin.file));
    }
    verify_file(&dir, &profile.message_catalog)?;
    Ok(VerifiedRuntime {
        dir,
        profile,
        expected_mapped,
    })
}

fn reject_arithmetic_overrides() -> Result<()> {
    // These oneMKL knobs can change dispatch, numerical reproducibility or
    // threading. The first profile was qualified without them. Do not mutate
    // the caller's process environment or silently ignore an explicit override.
    for name in [
        "MKL_CBWR",
        "MKL_ENABLE_INSTRUCTIONS",
        "MKL_DEBUG_CPU_TYPE",
        "MKL_THREADING_LAYER",
        "MKL_INTERFACE_LAYER",
        "MKL_DYNAMIC",
        "MKL_NUM_THREADS",
        "MKL_DOMAIN_NUM_THREADS",
        "MKL_NUM_STRIPES",
        "MKL_ENABLE_AVX512",
    ] {
        ensure!(
            std::env::var_os(name).is_none(),
            "unsupported MKL override for qualified v4 frontend: {name}"
        );
    }
    Ok(())
}

#[repr(C)]
#[derive(Debug, Default)]
struct MklStatus {
    native_status: i64,
    cleanup_status: i64,
    stage: i32,
    prior_threads: i32,
    effective_threads: i32,
    restored_previous: i32,
    descriptor_released: i32,
}
type AbiFn = unsafe extern "C" fn() -> u32;
type VersionFn = unsafe extern "C" fn(*mut c_char, usize) -> i32;
type ForwardFn =
    unsafe extern "C" fn(*mut f32, usize, *mut f32, usize, libc::c_long, *mut MklStatus) -> i32;

struct Functions {
    forward: ForwardFn,
}

/// Public prefix of GNU `struct link_map` from `<link.h>`; later private fields
/// are not accessed. Only GNU Linux x86-64 compiles this module.
#[repr(C)]
struct LinkMapPrefix {
    l_addr: usize,
    l_name: *const c_char,
    l_ld: *mut c_void,
    l_next: *mut LinkMapPrefix,
    l_prev: *mut LinkMapPrefix,
}

fn last_loader_error() -> String {
    // SAFETY: dlerror returns a thread-local, NUL-terminated string valid
    // until the next loader call; copy it immediately.
    unsafe {
        let error = libc::dlerror();
        if error.is_null() {
            "unknown dynamic loader error".to_owned()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    }
}

fn symbol(handle: usize, name: &CStr) -> Result<*mut c_void> {
    // SAFETY: handle is an owned live namespace handle, never closed. Clear
    // and inspect the thread-local loader error around dlsym.
    unsafe {
        libc::dlerror();
        let address = libc::dlsym(handle as *mut c_void, name.as_ptr());
        let error = libc::dlerror();
        if !error.is_null() {
            bail!(
                "MKL shim symbol lookup failed: {}",
                CStr::from_ptr(error).to_string_lossy()
            );
        }
        ensure!(!address.is_null(), "missing MKL shim symbol");
        Ok(address)
    }
}

fn namespace_paths(handle: usize) -> Result<Vec<PathBuf>> {
    let mut namespace = libc::LM_ID_BASE;
    let mut map: *mut LinkMapPrefix = std::ptr::null_mut();
    // SAFETY: dlinfo's documented GNU outputs match Lmid_t and a pointer to
    // the public link_map prefix. The live isolated handle is process-owned.
    // Only our serialized calls can cause additions to this namespace.
    unsafe {
        ensure!(
            libc::dlinfo(
                handle as *mut c_void,
                libc::RTLD_DI_LMID,
                (&mut namespace as *mut libc::Lmid_t).cast()
            ) == 0
                && namespace != libc::LM_ID_BASE,
            "MKL loader namespace is not isolated"
        );
        ensure!(
            libc::dlinfo(
                handle as *mut c_void,
                libc::RTLD_DI_LINKMAP,
                (&mut map as *mut *mut LinkMapPrefix).cast()
            ) == 0
                && !map.is_null(),
            "cannot inspect MKL namespace link map"
        );
        for _ in 0..64 {
            if (*map).l_prev.is_null() {
                break;
            }
            map = (*map).l_prev;
        }
        ensure!(
            (*map).l_prev.is_null(),
            "unexpected MKL namespace link-map size"
        );
        let mut paths = Vec::new();
        while !map.is_null() {
            ensure!(
                paths.len() < 64 && !(*map).l_name.is_null(),
                "unexpected MKL namespace link-map size or name"
            );
            let name = CStr::from_ptr((*map).l_name).to_bytes();
            ensure!(!name.is_empty(), "unexpected empty link-map path");
            paths.push(PathBuf::from(OsStr::from_bytes(name)));
            map = (*map).l_next;
        }
        Ok(paths)
    }
}

fn verify_namespace_paths(paths: &[PathBuf], expected: &BTreeSet<PathBuf>) -> Result<()> {
    let mut actual = BTreeSet::new();
    for path in paths {
        ensure!(
            path.is_absolute(),
            "unexpected non-absolute MKL namespace path"
        );
        let canonical = path.canonicalize()?;
        if expected.contains(&canonical) {
            ensure!(
                actual.insert(canonical),
                "duplicate MKL artifact in namespace"
            );
        } else {
            let name = path
                .file_name()
                .and_then(OsStr::to_str)
                .context("invalid system library name")?;
            ensure!(
                SYSTEM_NAMES.contains(&name)
                    && (canonical.starts_with("/usr/lib") || canonical.starts_with("/lib")),
                "unexpected library in isolated MKL namespace"
            );
        }
    }
    ensure!(
        &actual == expected,
        "isolated MKL namespace does not contain exact verified closure"
    );
    Ok(())
}

fn verify_namespace(handle: usize, expected: &BTreeSet<PathBuf>) -> Result<()> {
    verify_namespace_paths(&namespace_paths(handle)?, expected)
}

fn check_abi(actual: u32, expected: u32) -> Result<()> {
    ensure!(actual == expected, "unsupported MKL bridge ABI");
    Ok(())
}
fn check_version(actual: &str, expected: &str) -> Result<()> {
    ensure!(actual == expected, "unqualified MKL version");
    Ok(())
}

fn load_functions(handle: usize, verified: &VerifiedRuntime) -> Result<Functions> {
    verify_namespace(handle, &verified.expected_mapped)?;
    // SAFETY: exact pinned shim bytes provide these fixed, documented C ABI
    // exports. dlopen handles and symbol addresses remain live forever; POSIX
    // permits conversion of dlsym's result to the corresponding function pointer.
    let abi: AbiFn = unsafe { std::mem::transmute(symbol(handle, c"gigastt_v4_mkl_abi")?) };
    let version: VersionFn =
        unsafe { std::mem::transmute(symbol(handle, c"gigastt_v4_mkl_version")?) };
    let forward: ForwardFn =
        unsafe { std::mem::transmute(symbol(handle, c"gigastt_v4_mkl_forward")?) };
    // SAFETY: abi has no arguments; version receives the exact writable 512-byte buffer.
    check_abi(unsafe { abi() }, verified.profile.shim_abi)?;
    let mut buffer = [0u8; 512];
    ensure!(
        unsafe { version(buffer.as_mut_ptr().cast(), buffer.len()) } == 0,
        "MKL version query failed"
    );
    let actual = CStr::from_bytes_until_nul(&buffer)?.to_str()?;
    check_version(actual, &verified.profile.version)?;
    Ok(Functions { forward })
}

pub(super) struct MklRuntime {
    handle: usize,
    expected_mapped: BTreeSet<PathBuf>,
    functions: Functions,
    failed: Mutex<bool>,
}

struct RuntimeAttempt {
    dir: PathBuf,
    // Explicitly retain the native handle for process lifetime even when a
    // later ABI/namespace check failed. dlclose is not a reliable unload.
    _handle: Option<usize>,
    result: std::result::Result<MklRuntime, String>,
}

static RUNTIME: OnceLock<RuntimeAttempt> = OnceLock::new();

impl RuntimeAttempt {
    fn load(verified: VerifiedRuntime) -> Self {
        let shim = verified.dir.join(&verified.profile.shim.file);
        let cpath = match CString::new(shim.as_os_str().as_bytes()) {
            Ok(cpath) => cpath,
            Err(_) => {
                return Self {
                    dir: verified.dir,
                    _handle: None,
                    result: Err("invalid MKL shim path".to_owned()),
                };
            }
        };
        // SAFETY: explicit, trusted directory and every artifact's exact
        // bytes are verified before dlmopen; old RPATH=$ORIGIN, no global
        // loader scope or environment mutation. The local group is retained.
        let raw = unsafe {
            libc::dlmopen(
                libc::LM_ID_NEWLM,
                cpath.as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if raw.is_null() {
            return Self {
                dir: verified.dir,
                _handle: None,
                result: Err(last_loader_error()),
            };
        }
        let handle = raw as usize;
        let result = load_functions(handle, &verified)
            .map(|functions| MklRuntime {
                handle,
                expected_mapped: verified.expected_mapped,
                functions,
                failed: Mutex::new(false),
            })
            .map_err(|err| format!("{err:#}"));
        Self {
            dir: verified.dir,
            _handle: Some(handle),
            result,
        }
    }

    fn get_for(&self, dir: &Path) -> Result<&MklRuntime> {
        ensure!(
            self.dir == dir,
            "a different MKL runtime was already selected for this process"
        );
        self.result.as_ref().map_err(|error| {
            anyhow::anyhow!("MKL runtime initialization failed; restart required: {error}")
        })
    }
}

impl MklRuntime {
    pub(super) fn open(dir: &Path) -> Result<&'static Self> {
        reject_arithmetic_overrides()?;
        // Preflight path/hash failures do not create a namespace. Once a
        // native load is attempted, success or failure is permanent.
        let verified = validate_runtime_dir(dir)?;
        let canonical = verified.dir.clone();
        RUNTIME
            .get_or_init(|| RuntimeAttempt::load(verified))
            .get_for(&canonical)
    }

    pub(super) fn forward(
        &self,
        input: &mut AlignedFloats,
        output: &mut AlignedFloats,
        frames: usize,
    ) -> Result<()> {
        ensure!(
            (1..=3000).contains(&frames)
                && input.as_slice().len() == frames * 320
                && output.as_slice().len() == frames * 322,
            "invalid MKL transform geometry"
        );
        ensure!(
            input.as_slice().as_ptr().align_offset(64) == 0
                && output.as_slice().as_ptr().align_offset(64) == 0,
            "MKL transform buffers must be 64-byte aligned"
        );
        let mut failed = self
            .failed
            .lock()
            .map_err(|_| anyhow::anyhow!("MKL runtime lock poisoned"))?;
        ensure!(!*failed, "MKL runtime failed; restart required");
        let call = (|| {
            verify_namespace(self.handle, &self.expected_mapped)?;
            ensure!(
                input.guards_intact() && output.guards_intact(),
                "MKL FFT buffer guard changed before transform"
            );
            let mut status = MklStatus::default();
            let frames = libc::c_long::try_from(frames).context("MKL frame count overflow")?;
            let code = {
                let input = input.as_mut_slice();
                let output = output.as_mut_slice();
                // SAFETY: checked exact sizes/alignment and distinct mutable
                // Rust allocations; fixed ABI returns synchronously and never
                // retains pointers. Mutex serializes calls and namespace changes.
                unsafe {
                    (self.functions.forward)(
                        input.as_mut_ptr(),
                        input.len(),
                        output.as_mut_ptr(),
                        output.len(),
                        frames,
                        &mut status,
                    )
                }
            };
            ensure!(
                code == 0
                    && status.stage == 0
                    && status.native_status == 0
                    && status.cleanup_status == 0
                    && status.prior_threads >= 0
                    && status.effective_threads == 1
                    && status.restored_previous == 1
                    && status.descriptor_released == 1,
                "MKL DFTI failed: {status:?}"
            );
            ensure!(
                input.guards_intact() && output.guards_intact(),
                "MKL FFT buffer guard changed"
            );
            verify_namespace(self.handle, &self.expected_mapped)?;
            Ok(())
        })();
        if call.is_err() {
            *failed = true;
        }
        call?;
        // A successful, cleaned-up native call may overflow on extreme but
        // finite caller PCM. Reject that output, without treating it as a
        // native status/ownership/namespace failure or silently clipping it.
        ensure!(
            output.as_slice().iter().all(|x| x.is_finite()),
            "MKL DFTI returned non-finite or unwritten output"
        );
        Ok(())
    }
}

// Test-only positive symbol-conflict control. This deliberately installs a
// tiny exact fake DFTI symbol in the base namespace; production never does.
#[cfg(test)]
type FakeCreateFn = unsafe extern "C" fn(*mut c_void, i32, i32, libc::c_long, ...) -> libc::c_long;
#[cfg(test)]
type FakeCountFn = unsafe extern "C" fn() -> i32;

#[cfg(test)]
pub(super) struct FakeSymbolControl {
    handle: usize,
    function: FakeCreateFn,
    count: FakeCountFn,
    address: usize,
}

#[cfg(test)]
impl FakeSymbolControl {
    pub(super) fn open(path: &Path) -> Self {
        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: ignored parity test has verified the exact locally built
        // fake library bytes before this test-only base-namespace load.
        let raw = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
        assert!(!raw.is_null());
        let handle = raw as usize;
        let address = symbol(handle, c"DftiCreateDescriptor").unwrap();
        // SAFETY: exact fake library exports these declared C signatures.
        let function = unsafe { std::mem::transmute::<*mut c_void, FakeCreateFn>(address) };
        let count = unsafe {
            std::mem::transmute::<*mut c_void, FakeCountFn>(
                symbol(handle, c"gigastt_fake_calls").unwrap(),
            )
        };
        Self {
            handle,
            function,
            count,
            address: address as usize,
        }
    }
    pub(super) fn check(&self, expected: i32) {
        // SAFETY: RTLD_DEFAULT is the documented base lookup pseudo-handle;
        // this test executes the known fake function, which ignores inputs.
        assert_eq!(
            symbol(libc::RTLD_DEFAULT as usize, c"DftiCreateDescriptor").unwrap() as usize,
            self.address
        );
        assert_eq!(
            unsafe { (self.function)(std::ptr::null_mut(), 35, 33, 1, 320 as libc::c_long) },
            -777
        );
        assert_eq!(unsafe { (self.count)() }, expected);
    }
}

#[cfg(test)]
impl Drop for FakeSymbolControl {
    fn drop(&mut self) {
        // SAFETY: this test-only handle owns a live fake object; no use after drop.
        unsafe {
            libc::dlclose(self.handle as *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mkl_artifact_hash_size_and_regular_file_are_required() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.so");
        let mut h = Sha256::new();
        h.update(b"trusted");
        let pin = Pin {
            file: "test.so".into(),
            bytes: 7,
            sha256: hex_lower(&h.finalize()),
        };
        assert!(verify_file(dir.path(), &pin).is_err());
        std::fs::write(&path, b"trusted").unwrap();
        assert!(verify_file(dir.path(), &pin).is_ok());
        std::fs::write(&path, b"trusteD").unwrap();
        assert!(verify_file(dir.path(), &pin).is_err());
        std::fs::write(&path, b"short").unwrap();
        assert!(verify_file(dir.path(), &pin).is_err());
        std::fs::remove_file(&path).unwrap();
        let other = dir.path().join("other");
        std::fs::write(&other, b"trusted").unwrap();
        std::os::unix::fs::symlink(other, path).unwrap();
        assert!(verify_file(dir.path(), &pin).is_err());
    }

    #[test]
    fn test_mkl_wrong_abi_and_runtime_version_are_rejected() {
        let profile: Profile = serde_json::from_str(PROFILE).unwrap();
        assert!(check_abi(profile.shim_abi, profile.shim_abi).is_ok());
        assert!(check_abi(profile.shim_abi + 1, profile.shim_abi).is_err());
        assert!(check_version(&profile.version, &profile.version).is_ok());
        assert!(check_version("2025.0", &profile.version).is_err());
    }

    #[test]
    fn test_mkl_status_c_layout_and_missing_symbol() {
        assert_eq!(std::mem::size_of::<MklStatus>(), 40);
        // SAFETY: NULL requests the live main executable handle, not any
        // vendor library. This is a model-free missing-symbol test only.
        let handle = unsafe { libc::dlopen(std::ptr::null(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        assert!(!handle.is_null());
        assert!(symbol(handle as usize, c"gigastt_v4_mkl_missing_test_symbol").is_err());
        // SAFETY: the handle was just opened and no symbol is retained.
        assert_eq!(unsafe { libc::dlclose(handle) }, 0);
    }

    #[test]
    fn test_mkl_namespace_rejects_unknown_paths_without_loading() {
        let expected = BTreeSet::from([PathBuf::from("/not-a-real-library")]);
        assert!(verify_namespace_paths(&[], &expected).is_err());
        assert!(verify_namespace_paths(&[PathBuf::from("relative.so")], &expected).is_err());
    }
    #[test]
    fn test_mkl_failed_attempt_cannot_reload_another_directory() {
        let attempt = RuntimeAttempt {
            dir: PathBuf::from("/first"),
            _handle: None,
            result: Err("fake load failure".into()),
        };
        assert!(
            attempt
                .get_for(Path::new("/first"))
                .err()
                .unwrap()
                .to_string()
                .contains("restart required")
        );
        assert!(
            attempt
                .get_for(Path::new("/other"))
                .err()
                .unwrap()
                .to_string()
                .contains("different")
        );
    }
}
