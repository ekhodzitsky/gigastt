//! Stage files beside their destinations so publication can refuse replacement.

use super::{BatchOptions, FileSignature, output_path_for};
use anyhow::Context;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

pub(super) struct Directories {
    pub input: PathBuf,
    pub output: PathBuf,
    pub archive: Option<PathBuf>,
}

impl Directories {
    pub(super) fn prepare(opts: &BatchOptions) -> anyhow::Result<Self> {
        let prepare_destination = |dir: &Path| -> anyhow::Result<PathBuf> {
            let prepare = || {
                fs::create_dir_all(dir)?;
                dir.canonicalize()
            };
            prepare().with_context(|| format!("failed to prepare {}", dir.display()))
        };
        Ok(Self {
            input: opts.input_dir.canonicalize().with_context(|| {
                format!("input directory not found: {}", opts.input_dir.display())
            })?,
            output: prepare_destination(&opts.output_dir)?,
            archive: opts
                .move_to
                .as_deref()
                .map(prepare_destination)
                .transpose()?,
        })
    }
}

pub(super) struct Destinations {
    pub reports: Vec<PathBuf>,
    pub archive: Option<PathBuf>,
}

impl Destinations {
    pub(super) fn plan(
        source: &Path,
        output: &Path,
        formats: &[super::ExportFormat],
        archive: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let archive = archive
            .map(|dir| {
                let name = source
                    .file_name()
                    .with_context(|| format!("no file name: {}", source.display()))?;
                Ok::<_, anyhow::Error>(dir.join(name))
            })
            .transpose()?;
        let reports = formats
            .iter()
            .map(|format| output_path_for(source, output, format.extension()))
            .collect();
        Ok(Self { reports, archive })
    }

    pub(super) fn paths(&self) -> impl Iterator<Item = &Path> {
        self.reports
            .iter()
            .chain(&self.archive)
            .map(PathBuf::as_path)
    }

    pub(super) fn ensure_unused(&self) -> anyhow::Result<()> {
        self.paths().try_for_each(ensure_unused)
    }
}

/// Treat directories and dangling symlinks as occupied as well as regular files.
pub(super) fn ensure_unused(path: &Path) -> anyhow::Result<()> {
    let err = match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => error,
        Ok(_) => io::Error::new(io::ErrorKind::AlreadyExists, "destination already exists"),
    };
    Err(err).with_context(|| format!("cannot create {}", path.display()))
}

fn stage(destination: &Path) -> anyhow::Result<NamedTempFile> {
    let parent = destination
        .parent()
        .with_context(|| format!("no parent directory: {}", destination.display()))?;
    let mut builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Match File::create permissions, including the process umask.
        builder.permissions(fs::Permissions::from_mode(0o666));
    }
    builder
        .prefix(".gigastt-")
        .tempfile_in(parent)
        .with_context(|| format!("failed to stage {}", destination.display()))
}

pub(super) fn stage_report(destination: &Path, contents: &str) -> anyhow::Result<NamedTempFile> {
    let mut file = stage(destination)?;
    file.write_all(contents.as_bytes())
        .with_context(|| format!("failed to write {}", destination.display()))?;
    Ok(file)
}

pub(super) fn publish(staged: NamedTempFile, destination: &Path) -> anyhow::Result<()> {
    staged
        .persist_noclobber(destination)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to publish {}", destination.display()))?;
    Ok(())
}

pub(super) fn move_file(
    source: &Path,
    destination: &Path,
    signature: FileSignature,
) -> anyhow::Result<()> {
    signature.ensure_unchanged(source)?;
    // Linking a regular file supplies a fast, exclusive destination on the
    // same filesystem. Copy the contents when links are unavailable or the
    // source is a symlink, whose relative target can break when relocated.
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("failed to inspect source {}", source.display()))?;
    if metadata.is_file() {
        match fs::hard_link(source, destination) {
            Ok(()) => return remove_source(source, signature),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                return Err(error)
                    .with_context(|| format!("failed to archive {}", destination.display()));
            }
            Err(_) => {}
        }
    }
    let mut source_file = fs::File::open(source)
        .with_context(|| format!("failed to open source {}", source.display()))?;
    let mut staged = stage(destination)?;
    io::copy(&mut source_file, staged.as_file_mut()).with_context(|| {
        format!(
            "failed to copy {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    staged
        .as_file()
        .set_permissions(source_file.metadata()?.permissions())
        .with_context(|| format!("failed to set permissions on {}", destination.display()))?;
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("failed to sync {}", destination.display()))?;
    signature.ensure_unchanged(source)?;
    publish(staged, destination)?;
    remove_source(source, signature)
}

pub(super) fn remove_source(source: &Path, signature: FileSignature) -> anyhow::Result<()> {
    signature.ensure_unchanged(source)?;
    fs::remove_file(source).with_context(|| format!("failed to remove source {}", source.display()))
}
