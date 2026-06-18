// swift-tools-version: 6.2
import PackageDescription

// mlx-sea-raft-swift — SEA-RAFT optical flow for MLXEngine. ONE repo, TWO products:
//   • SEARAFTMLX — engine-agnostic Swift/MLX core (no MLXToolKit dep; usable standalone)
//   • MLXSEARAFT — the MLXEngine `opticalFlow` ModelPackage over that core
// Consolidated 2026-06-18: the former standalone `sea-raft-mlx-swift` core was folded in (archived).
let package = Package(
    name: "mlx-sea-raft-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "SEARAFTMLX", targets: ["SEARAFTMLX"]),
        .library(name: "MLXSEARAFT", targets: ["MLXSEARAFT"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.8.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "SEARAFTMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXSEARAFT",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "SEARAFTMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "SEARAFTMLXTests",
            dependencies: [
                "SEARAFTMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
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
