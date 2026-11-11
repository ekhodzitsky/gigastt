use std::{env, error::Error, io, path::PathBuf, process::Command};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var_os("CARGO_FEATURE_COREML").is_none()
    {
        return Ok(());
    }

    // Rust's -nodefaultlibs omits compiler-rt, which the static CoreML provider
    // needs for Objective-C @available checks (__isPlatformVersionAtLeast).
    // Ask the selected Xcode toolchain instead of hardcoding its version/path.
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=PATH");
    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", "clang", "--print-resource-dir"])
        .output()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("CoreML compiler-rt discovery: {error}"),
            )
        })?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "CoreML compiler-rt discovery failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }
    let resource = String::from_utf8(output.stdout)?;
    let resource = resource.trim();
    if resource.is_empty() || resource.contains(['\n', '\r']) {
        return Err(io::Error::other("CoreML clang returned an invalid resource directory").into());
    }
    let directory = PathBuf::from(resource).join("lib/darwin");
    let archive = directory.join("libclang_rt.osx.a");
    if !archive.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("CoreML compiler runtime missing: {}", archive.display()),
        )
        .into());
    }
    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rustc-link-search=native={}", directory.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
    Ok(())
}
