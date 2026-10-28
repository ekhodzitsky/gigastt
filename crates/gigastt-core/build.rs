use std::{env, io, path::PathBuf, process::Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var_os("CARGO_FEATURE_COREML").is_none()
        || env::var_os("CARGO_FEATURE_ORT_LOAD_DYNAMIC").is_some()
    {
        return Ok(());
    }
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    println!("cargo:rerun-if-env-changed=TOOLCHAINS");

    // ONNX Runtime 1.31's static CoreML objects call Clang's availability
    // builtin (__isPlatformVersionAtLeast). Rust links with -nodefaultlibs,
    // so Clang does not automatically add its compiler runtime archive.
    // Use the selected Apple toolchain's implementation, not an OS-version stub.
    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", "clang", "--print-resource-dir"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "cannot locate the Apple Clang runtime: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    let archive = PathBuf::from(String::from_utf8(output.stdout)?.trim())
        .join("lib/darwin/libclang_rt.osx.a");
    if !archive.is_absolute() || !archive.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Apple Clang did not provide libclang_rt.osx.a: {}",
                archive.display()
            ),
        )
        .into());
    }
    let directory = archive
        .parent()
        .ok_or_else(|| io::Error::other("Apple Clang runtime archive has no parent directory"))?;
    println!("cargo:rustc-link-search=native={}", directory.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
    Ok(())
}
