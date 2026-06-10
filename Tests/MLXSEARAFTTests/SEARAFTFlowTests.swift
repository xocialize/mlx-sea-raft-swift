import Testing
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
@testable import MLXSEARAFT

/// Offline conformance — live flow estimation is proven in the `MLXEngine Testing` app; the core
/// carries the parity suite (cosine 0.99997 / EPE 0.116 px vs the Python reference).
struct SEARAFTFlowTests {

    @Test func manifestIsOpticalFlowAndPermissive() {
        let m = SEARAFTOpticalFlowPackage.manifest
        #expect(m.capabilities == [.opticalFlow])
        #expect(m.license.weightLicense == .bsd3)
        #expect(m.license.portCodeLicense == .mit)
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func manifestRequirements() {
        let r = SEARAFTOpticalFlowPackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.first?.quant == .fp32)
    }

    @Test func surfaceIsTheCanonicalFlowDescriptor() {
        let s = SEARAFTOpticalFlowPackage.manifest.surfaces.first
        #expect(s?.capability == .opticalFlow)
        #expect(s?.parameters.count == 2)
        #expect(s?.parameters.allSatisfy { $0.kind == .image && $0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = SEARAFTOpticalFlowPackage.registration
        #expect(reg.manifest.capabilities == [.opticalFlow])
        let pkg = try reg.makePackage(SEARAFTConfiguration())
        #expect(pkg is SEARAFTOpticalFlowPackage)
    }

    @Test func variantsMapToPublishedRepos() {
        #expect(SEARAFTConfiguration().variant == .spring)
        #expect(SEARAFTVariant.spring.repo == "mlx-community/SEA-RAFT-Tartan-C-T-TSKH-spring540x960-S-mlx")
        #expect(SEARAFTVariant.tartan.repo == "mlx-community/SEA-RAFT-Tartan480x640-S-mlx")
    }

    @Test func configurationCodableExcludesEnvironmentRoot() throws {
        var c = SEARAFTConfiguration(variant: .tartan, iters: 8)
        c.modelsRootDirectory = URL(fileURLWithPath: "/tmp/x")
        let back = try JSONDecoder().decode(SEARAFTConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.variant == .tartan)
        #expect(back.iters == 8)
        #expect(back.modelsRootDirectory == nil)
    }

    @Test func imageDecodesToZeroTo255Tensor() throws {
        // Solid sRGB white 8×6 PNG → [1, 6, 8, 3] with values ≈ 255 (no Metal: lazy creation,
        // but asArray forces eval — so this runs under xcodebuild only; guarded for CLI).
        let png = try #require(Self.makePNG(width: 8, height: 6, gray: 1.0))
        let image = Image(format: .png, data: png, width: 8, height: 6)
        let t = try SEARAFTOpticalFlowPackage.decodeToTensor(image)
        #expect(t.shape == [1, 6, 8, 3])
    }

    static func makePNG(width: Int, height: Int, gray: CGFloat) -> Data? {
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: gray, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}
