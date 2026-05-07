fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();

    if let Ok(bindings) = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
    {
        let out =
            std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("include");
        std::fs::create_dir_all(&out).ok();
        bindings.write_to_file(out.join("gigastt.h"));
    }
}
