'use strict'
// postinstall: download ONLY the native addon for THIS platform from the
// GitHub release, into the package directory. Keeps the npm package a single
// JS-only package while installing just one ~47 MB binary per machine.
//
// Node built-ins only (runs before the consumer's deps are available). Honors
// GIGASTT_NODE_TAG to override the release tag (default `node-v<version>`).
const fs = require('fs')
const path = require('path')
const https = require('https')

const { version } = require('./package.json')

function platformKey() {
  const { platform, arch } = process
  if (platform === 'darwin' && arch === 'arm64') return 'darwin-arm64'
  if (platform === 'linux' && arch === 'x64') return 'linux-x64-gnu'
  if (platform === 'linux' && arch === 'arm64') return 'linux-arm64-gnu'
  if (platform === 'win32' && arch === 'x64') return 'win32-x64-msvc'
  return null
}

const key = platformKey()
if (!key) {
  console.error(
    `gigastt: no prebuilt binary for ${process.platform}-${process.arch}; ` +
      `build from source (https://github.com/ekhodzitsky/gigastt).`
  )
  // Don't hard-fail install on unsupported platforms; loading will error clearly.
  process.exit(0)
}

const file = `gigastt.${key}.node`
const dest = path.join(__dirname, file)
if (fs.existsSync(dest)) {
  console.log(`gigastt: ${file} already present`)
  process.exit(0)
}

const tag = process.env.GIGASTT_NODE_TAG || `node-v${version}`
const url = `https://github.com/ekhodzitsky/gigastt/releases/download/${tag}/${file}`

function download(u, redirects, cb) {
  https
    .get(u, { headers: { 'user-agent': 'gigastt-installer' } }, (res) => {
      const { statusCode, headers } = res
      if ([301, 302, 303, 307, 308].includes(statusCode) && headers.location && redirects < 6) {
        res.resume()
        return download(headers.location, redirects + 1, cb)
      }
      if (statusCode !== 200) {
        res.resume()
        return cb(new Error(`HTTP ${statusCode} for ${u}`))
      }
      const tmp = `${dest}.download`
      const out = fs.createWriteStream(tmp)
      res.pipe(out)
      out.on('finish', () => out.close(() => {
        fs.renameSync(tmp, dest)
        cb()
      }))
      out.on('error', cb)
    })
    .on('error', cb)
}

console.log(`gigastt: downloading ${file} from ${tag} ...`)
download(url, 0, (err) => {
  if (err) {
    // Don't hard-fail the install (this also runs during local dev/CI before
    // the binary is built or the release exists). loader.js raises a clear
    // error at require-time if the binary is genuinely missing.
    console.warn(`gigastt: could not download ${file}: ${err.message}`)
    console.warn(`gigastt: tried ${url}`)
    process.exit(0)
  }
  console.log(`gigastt: installed ${file}`)
})
