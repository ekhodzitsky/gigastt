'use strict'
// Single-package loader: resolves the one native addon for THIS platform.
// The matching `gigastt.<platform>.node` is fetched at install time by
// install.js (postinstall) from the GitHub release; it is not bundled, so only
// one binary is downloaded per machine (not all platforms).
const fs = require('fs')
const path = require('path')

function platformKey() {
  const { platform, arch } = process
  if (platform === 'darwin' && arch === 'arm64') return 'darwin-arm64'
  if (platform === 'linux' && arch === 'x64') return 'linux-x64-gnu'
  if (platform === 'linux' && arch === 'arm64') return 'linux-arm64-gnu'
  if (platform === 'win32' && arch === 'x64') return 'win32-x64-msvc'
  throw new Error(
    `gigastt: unsupported platform ${platform}-${arch}. ` +
      `Prebuilt binaries: darwin-arm64, linux-x64-gnu, linux-arm64-gnu, win32-x64-msvc. ` +
      `See https://github.com/ekhodzitsky/gigastt to build from source.`
  )
}

const binary = path.join(__dirname, `gigastt.${platformKey()}.node`)

if (!fs.existsSync(binary)) {
  throw new Error(
    `gigastt: native binary not found at ${binary}. ` +
      `The postinstall download was likely skipped (e.g. \`npm install --ignore-scripts\`). ` +
      `Run \`node ${path.join(__dirname, 'install.js')}\` to fetch it.`
  )
}

module.exports = require(binary)
module.exports.platformKey = platformKey
