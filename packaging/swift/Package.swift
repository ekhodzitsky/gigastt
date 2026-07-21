// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GigaSTT",
    platforms: [
        .iOS(.v15),
        .macOS(.v13)
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
            url: "https://github.com/ekhodzitsky/gigastt/releases/download/v2.13.0/GigasttFFI.xcframework.zip",
            checksum: "02a0c1989664b511c77cd5e85e09b181c760c53df3fc2bea807c341de4e17064"
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
            dependencies: ["GigasttFFI"],
            // ONNX Runtime inside the static archive is C++; its `-lc++` link
            // directive is emitted by cargo for the Rust link only and does not
            // propagate through the xcframework, so the consumer link must add
            // libc++ explicitly or the final app fails with undefined
            // `std::__1::*` symbols.
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        )
    ]
)
