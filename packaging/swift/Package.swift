// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GigaSTT",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(name: "GigaSTT", targets: ["GigaSTT"])
    ],
    targets: [
        // Prebuilt C-ABI static library, shipped as an xcframework attached to
        // a GitHub release by .github/workflows/ios-xcframework.yml.
        //
        // The `url:` and `checksum:` below are rewritten automatically by that
        // workflow on every release it runs for — do not edit them by hand.
        .binaryTarget(
            name: "GigasttFFI",
            url: "https://github.com/ekhodzitsky/gigastt/releases/download/v2.10.0/GigasttFFI.xcframework.zip",
            checksum: "cf1d3cc12ebb2c2353797a762f53f5b0add48a147b07d7b4e16f696a177e4ba6"
        ),
        // --- Local development -------------------------------------------------
        // To build against a locally produced xcframework instead of the
        // released zip, comment out the `.binaryTarget(... url: ...)` above and
        // uncomment the path form below, then drop GigasttFFI.xcframework next
        // to this Package.swift:
        //
        // .binaryTarget(
        //     name: "GigasttFFI",
        //     path: "GigasttFFI.xcframework"
        // ),
        // -----------------------------------------------------------------------
        .target(
            name: "GigaSTT",
            dependencies: ["GigasttFFI"]
        )
    ]
)
