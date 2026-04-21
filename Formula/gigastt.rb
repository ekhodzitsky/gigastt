# Homebrew formula for gigastt.
#
# Install with:
#   brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
#   brew install gigastt
#
# The `sha256` values below are pinned to the v<version> release tarballs.
# They are refreshed automatically by the `.github/workflows/homebrew.yml`
# workflow after every successful `release.yml` run — do not hand-edit
# unless you are backfilling a release that predated that automation.

class Gigastt < Formula
  desc "On-device Russian speech recognition server powered by GigaAM v3"
  homepage "https://github.com/ekhodzitsky/gigastt"
  version "0.9.4"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v0.9.4/gigastt-0.9.4-aarch64-apple-darwin.tar.gz"
      sha256 "325d13d398bc6e1a6bc5e15dfcf917ea426db1d48c9ff7b9397cb6b6569dba2f"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v0.9.4/gigastt-0.9.4-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "653b77a0574ad2b9d68df1ed46589e7eace5174d179e8098087b3a7e98f622c5"
    end
  end

  def install
    bin.install "gigastt"
  end

  def caveats
    <<~EOS
      The GigaAM v3 model (~850 MB) is downloaded on first run into
      ~/.gigastt/models. An INT8-quantized encoder is produced automatically
      (~2 min one-time). Disable with `--skip-quantize` or
      `GIGASTT_SKIP_QUANTIZE=1`.

      Quick start:
        gigastt download         # fetches model + quantizes
        gigastt serve            # starts STT server on 127.0.0.1:9876

      Homepage: https://github.com/ekhodzitsky/gigastt
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/gigastt --version")
  end
end
