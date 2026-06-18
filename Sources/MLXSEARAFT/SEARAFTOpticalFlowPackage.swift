import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import MLXToolKit
import MLX
import Hub
import SEARAFTMLX

/// Errors at the SEA-RAFT package boundary.
public enum SEARAFTPackageError: Error {
    case imageDecodeFailed(String)
    case dimensionMismatch(String)
}

/// An MLXEngine `opticalFlow` package over **SEA-RAFT** (Wang/Lipson/Deng, ECCV 2024, BSD-3) —
/// dense per-pixel motion between a frame pair. The temporal building block of the visual
/// optimization tier: backward-warping for temporal-consistent enhancement, and motion features
/// for the pipeline planner.
///
/// A thin conformance wrapper over the parity-locked `SEARAFTMLX` core (cosine 0.99997 / EPE
/// 0.116 px vs the Python reference, itself cosine-1.0 locked vs PyTorch).
@InferenceActor
public final class SEARAFTOpticalFlowPackage: ModelPackage {
    public typealias Configuration = SEARAFTConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // SEA-RAFT code + weights: BSD-3-Clause (Princeton VL Lab; per-checkpoint
            // confirmation pending at princeton-vl/SEA-RAFT#31, linked from the weight cards).
            license: LicenseDeclaration(weightLicense: .bsd3, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/SEA-RAFT-Tartan-C-T-TSKH-spring540x960-S-mlx",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // ~34 MB fp32 weights; the all-pairs correlation volume dominates the working
                // set ((H/8·W/8)² — budget for 540×960-class inference, the upstream protocol).
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 4_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                OpticalFlowContract.descriptor(
                    name: "sea-raft-flow",
                    summary: "SEA-RAFT dense optical flow between two frames (pixel displacements; iteration knob for quality/speed)."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: SEARAFT?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }
        let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
        let dir = try await hub.snapshot(from: Hub.Repo(id: configuration.variant.repo),
                                         matching: ["model.safetensors"]) { progress, speed in
            WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
        }
        let m = SEARAFT(.s)
        try m.loadWeights(from: dir.appendingPathComponent("model.safetensors"))
        m.train(false)   // BatchNorm running stats
        model = m
    }

    public func unload() async {
        model = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .opticalFlow,
              let req = request as? OpticalFlowRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let a = try Self.decodeToTensor(req.image0)
        let b = try Self.decodeToTensor(req.image1)
        guard a.shape == b.shape else {
            throw SEARAFTPackageError.dimensionMismatch("\(a.shape) vs \(b.shape)")
        }

        let flow = model(a, b, iters: configuration.iters)   // [1, H, W, 2] px
        MLX.eval(flow)

        let H = flow.shape[1], W = flow.shape[2]
        let uv = flow[0].asType(.float32).asArray(Float.self)  // row-major (u, v) interleaved
        return OpticalFlowResponse(flow: FlowField(width: W, height: H, uv: uv))
    }

    // MARK: - Image decode

    /// Decode a canonical `Image` (.png/.jpeg) to `[1, H, W, 3]` RGB floats in 0…255.
    nonisolated static func decodeToTensor(_ image: Image) throws -> MLXArray {
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw SEARAFTPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        let bytesPerRow = w * 4
        var raw = [UInt8](repeating: 0, count: h * bytesPerRow)
        try raw.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: bytesPerRow, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                throw SEARAFTPackageError.imageDecodeFailed("CGContext")
            }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var rgb = [Float](repeating: 0, count: h * w * 3)
        for p in 0..<(h * w) {
            rgb[p * 3 + 0] = Float(raw[p * 4 + 0])
            rgb[p * 3 + 1] = Float(raw[p * 4 + 1])
            rgb[p * 3 + 2] = Float(raw[p * 4 + 2])
        }
        return MLXArray(rgb, [1, h, w, 3])
    }
}

extension SEARAFTOpticalFlowPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(SEARAFTOpticalFlowPackage.self)
    }
}
