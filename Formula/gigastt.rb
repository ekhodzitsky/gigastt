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
  version "1.0.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v1.0.0/gigastt-1.0.0-aarch64-apple-darwin.tar.gz"
      sha256 "abfe5430033adb10865000edd96918ba661d1cc9d30e40b8d2b15683348395d3"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v1.0.0/gigastt-1.0.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "6fd20de808dd7e469e1d9f56c3a92bdd3c17dc7ae3c134faa544a465d4e1bb2b"
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
