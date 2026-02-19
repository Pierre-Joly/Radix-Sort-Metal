// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "RadixSortMetal",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "RadixSortMetal",
            targets: ["RadixSortMetal"]
        )
    ],
    targets: [
        .target(
            name: "RadixSortMetal",
            path: "RadixSortMetal",
            exclude: [
                "RadixSortMetal.metal"
            ],
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "RadixSortMetalTests",
            dependencies: ["RadixSortMetal"],
            path: "RadixSortMetalTests"
        ),
        .executableTarget(
            name: "RadixSortBenchmark",
            dependencies: ["RadixSortMetal"],
            path: "Benchmarks/RadixSortBenchmark"
        )
    ]
)
