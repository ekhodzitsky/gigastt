# Homebrew formula for gigastt.
#
# Install with:
#   brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
#   brew install gigastt
#
# The `sha256` values below are pinned to the v<version> release tarballs.
# Updates are proposed by `.github/workflows/homebrew.yml` after successful
# version-tag releases, then reviewed and merged through a pull request.

class Gigastt < Formula
  desc "On-device Russian speech recognition server powered by GigaAM v3"
  homepage "https://github.com/ekhodzitsky/gigastt"
  version "2.22.0"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.22.0/gigastt-2.22.0-aarch64-apple-darwin.tar.gz"
      sha256 "6cbcc42b4aa76feaf9c4df88b3de5e374d681e28702bb70c9fa38cb0329e2368"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.22.0/gigastt-2.22.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "cd790191a600df48aa5c56e124953322af89df4bbd60d2c980436b0597c46194"
    elsif Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.22.0/gigastt-2.22.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "8caef0df3be41db7cb2375b0922dbe8b55d3ab2c3b8bd049eac8e573296cc224"
    end
  end

  def install
    bin.install "gigastt"
  end

  def caveats
    <<~EOS
      The GigaAM v3 INT8 model (~225 MB) is downloaded on first run into
      ~/.gigastt/models (lean prequantized path; no FP32 step).

      Quick start:
        gigastt download         # lean INT8 bundle (~225 MB)
        gigastt serve            # starts STT server on 127.0.0.1:9876

      Homepage: https://github.com/ekhodzitsky/gigastt
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/gigastt --version")
  end
end
