// swift-tools-version: 6.2
import PackageDescription

// mlx-sea-raft-swift — the MLXEngine `opticalFlow` package over SEA-RAFT (ECCV 2024).
// The temporal building block of the visual optimization tier: dense per-pixel motion between
// frame pairs (warping for temporal consistency, planner motion features). Thin conformance
// layer over the parity-locked sea-raft-mlx-swift core. Module is `MLXSEARAFT`.
let package = Package(
    name: "mlx-sea-raft-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXSEARAFT", targets: ["MLXSEARAFT"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.8.0"),
        .package(url: "https://github.com/xocialize/sea-raft-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXSEARAFT",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "SEARAFTMLX", package: "sea-raft-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXSEARAFTTests",
            dependencies: [
                "MLXSEARAFT",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
