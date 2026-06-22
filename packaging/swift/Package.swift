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
        // After the first workflow run, paste the printed `url:` and
        // `checksum:` into the binary target below.
        .binaryTarget(
            name: "GigasttFFI",
            url: "REPLACE_WITH_URL",
            checksum: "REPLACE_WITH_CHECKSUM"
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
