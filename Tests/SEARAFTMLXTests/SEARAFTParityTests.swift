import Testing
import Foundation
import MLX
import MLXNN
@testable import SEARAFTMLX

/// Elementwise parity vs the Python sea-raft-mlx reference (itself parity-locked vs PyTorch:
/// cosine 1.0000 / EPE 1e-4 px). Requires the staged metallib — run via `xcodebuild test`.
/// Fixture + weights paths are environment-specific; tests skip when absent.
struct SEARAFTParityTests {
    static let fixtureURL = URL(fileURLWithPath: "/tmp/searaft_parity.safetensors")
    static let weightsURL = URL(fileURLWithPath: "/tmp/searaft_mlx_weights.safetensors")

    @Test func finalFlowMatchesPythonReference() throws {
        try #require(FileManager.default.fileExists(atPath: Self.fixtureURL.path))
        try #require(FileManager.default.fileExists(atPath: Self.weightsURL.path))

        let fixtures = try MLX.loadArrays(url: Self.fixtureURL)
        let model = SEARAFT(.s)
        try model.loadWeights(from: Self.weightsURL)
        model.train(false)   // BatchNorm must use running stats

        let flow = model(fixtures["img1"]!, fixtures["img2"]!).asType(.float32)
        MLX.eval(flow)

        let expected = fixtures["expected_final"]!.asType(.float32)
        #expect(flow.shape == expected.shape)

        let maxAbs = MLX.abs(flow - expected).max().item(Float.self)
        let dot = MLX.sum(flow * expected).item(Float.self)
        let n1 = MLX.sum(flow * flow).item(Float.self).squareRoot()
        let n2 = MLX.sum(expected * expected).item(Float.self).squareRoot()
        let cosine = dot / max(n1 * n2, 1e-12)
        let diff = flow - expected
        let epe = MLX.mean(MLX.sqrt(MLX.sum(diff * diff, axis: -1))).item(Float.self)
        print("[SEARAFT-PARITY] cosine=\(cosine) max_abs=\(maxAbs)px epe=\(epe)px")

        // Same-framework (MLX py CPU vs MLX swift GPU): allow Metal accumulation drift.
        #expect(cosine > 0.9999)
        #expect(epe < 0.25)
    }
}
