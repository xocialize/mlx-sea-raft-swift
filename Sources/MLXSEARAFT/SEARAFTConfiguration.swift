import Foundation
import MLXToolKit

/// Which published SEA-RAFT-S checkpoint to load.
public enum SEARAFTVariant: String, Codable, Sendable, CaseIterable {
    /// Full training schedule — best accuracy. Default.
    case spring
    /// TartanAir-stage only (CC-BY training data — the cleanest provenance chain).
    case tartan

    public var repo: String {
        switch self {
        case .spring: return "mlx-community/SEA-RAFT-Tartan-C-T-TSKH-spring540x960-S-mlx"
        case .tartan: return "mlx-community/SEA-RAFT-Tartan480x640-S-mlx"
        }
    }
}

/// Init-time configuration for `SEARAFTOpticalFlowPackage` (C9).
public struct SEARAFTConfiguration: PackageConfiguration, ModelStorable {
    public var variant: SEARAFTVariant
    /// Refinement iterations (the quality/speed knob; checkpoint-native default is 4).
    public var iters: Int
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(variant: SEARAFTVariant = .spring, iters: Int = 4, modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.iters = iters
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant, iters
    }
}
