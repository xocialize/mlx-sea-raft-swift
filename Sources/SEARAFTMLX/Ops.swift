//
//  Ops.swift
//  SEARAFTMLX
//
//  Hand-rolled NHWC spatial ops — isomorphic to sea-raft-mlx's ops.py (parity-locked there
//  vs torch). NOTE: the corr lookup uses torch's DEFAULT padding_mode='zeros' grid_sample —
//  a different variant than RIFE's border mode.
//

import Foundation
import MLX

/// Bilinear grid_sample (NHWC) with ZEROS padding — torch F.grid_sample default.
/// Out-of-range neighbours contribute 0 (per-corner validity mask).
/// - Parameters:
///   - input: `[B, H, W, C]`
///   - grid: `[B, gH, gW, 2]` normalized [-1, 1] (x, y)
public func gridSampleBilinearZeros(_ input: MLXArray, grid: MLXArray,
                                    alignCorners: Bool = true) -> MLXArray {
    let B = input.shape[0], H = input.shape[1], W = input.shape[2], C = input.shape[3]
    let gH = grid.shape[1], gW = grid.shape[2]

    let gx = grid[0..., 0..., 0..., 0]
    let gy = grid[0..., 0..., 0..., 1]
    let ix: MLXArray
    let iy: MLXArray
    if alignCorners {
        ix = (gx + 1) * 0.5 * Float(W - 1)
        iy = (gy + 1) * 0.5 * Float(H - 1)
    } else {
        ix = ((gx + 1) * Float(W) - 1) * 0.5
        iy = ((gy + 1) * Float(H) - 1) * 0.5
    }

    let x0 = floor(ix), y0 = floor(iy)
    let x1 = x0 + 1, y1 = y0 + 1
    let wx1 = ix - x0, wx0 = 1.0 - (ix - x0)
    let wy1 = iy - y0, wy0 = 1.0 - (iy - y0)

    let inputFlat = input.reshaped([B, H * W, C])

    func corner(_ xc: MLXArray, _ yc: MLXArray, _ w: MLXArray) -> MLXArray {
        let valid = (xc .>= 0) .&& (xc .<= Float(W - 1)) .&& (yc .>= 0) .&& (yc .<= Float(H - 1))
        let xs = clip(xc, min: 0, max: Float(W - 1)).asType(.int32)
        let ys = clip(yc, min: 0, max: Float(H - 1)).asType(.int32)
        var idx = (ys * Int32(W) + xs).reshaped([B, gH * gW, 1])
        idx = broadcast(idx, to: [B, gH * gW, C])
        let v = takeAlong(inputFlat, idx, axis: 1).reshaped([B, gH, gW, C])
        return v * (w * valid.asType(input.dtype)).expandedDimensions(axis: -1)
    }

    return corner(x0, y0, wy0 * wx0) + corner(x1, y0, wy0 * wx1)
        + corner(x0, y1, wy1 * wx0) + corner(x1, y1, wy1 * wx1)
}

private func sampleCoords(out: Int, inSize: Int, alignCorners: Bool) -> MLXArray {
    let dst = MLXArray(Array(0..<out).map { Float($0) })
    if alignCorners {
        let scale = out > 1 ? Float(inSize - 1) / Float(out - 1) : 0
        return dst * scale
    }
    let scale = Float(inSize) / Float(out)
    return (dst + 0.5) * scale - 0.5
}

private func bilinear1D(_ x: MLXArray, axis: Int, out: Int, alignCorners: Bool) -> MLXArray {
    let inSize = x.shape[axis]
    if inSize == out { return x }
    let src = sampleCoords(out: out, inSize: inSize, alignCorners: alignCorners)
    let i0f = floor(src)
    let w1 = src - i0f, w0 = 1.0 - (src - i0f)
    let i0 = clip(i0f, min: 0, max: Float(inSize - 1)).asType(.int32)
    let i1 = clip(i0f + 1, min: 0, max: Float(inSize - 1)).asType(.int32)
    let g0 = take(x, i0, axis: axis)
    let g1 = take(x, i1, axis: axis)
    var shape = [Int](repeating: 1, count: x.ndim)
    shape[axis] = out
    return g0 * w0.reshaped(shape) + g1 * w1.reshaped(shape)
}

/// Bilinear resize (NHWC) — F.interpolate(mode: "bilinear"). Degenerate targets clamp to 1.
public func interpolateBilinear(_ x: MLXArray, scaleFactor: Float,
                                alignCorners: Bool = false) -> MLXArray {
    let H = x.shape[1], W = x.shape[2]
    let oH = max(1, Int((Float(H) * scaleFactor).rounded()))
    let oW = max(1, Int((Float(W) * scaleFactor).rounded()))
    var out = bilinear1D(x, axis: 1, out: oH, alignCorners: alignCorners)
    out = bilinear1D(out, axis: 2, out: oW, alignCorners: alignCorners)
    return out
}

/// `[N, H, W, C] → [N, H, W, C, 9]` zero-padded 3×3 patches (p = ky*3 + kx, torch unfold order).
public func unfold3x3(_ x: MLXArray) -> MLXArray {
    let N = x.shape[0], H = x.shape[1], W = x.shape[2]
    let xp = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((1, 1)), IntOrPair((1, 1)), IntOrPair((0, 0))])
    var patches = [MLXArray]()
    patches.reserveCapacity(9)
    for ky in 0..<3 {
        for kx in 0..<3 {
            patches.append(xp[0..., ky..<(ky + H), kx..<(kx + W), 0...])
        }
    }
    return stacked(patches, axis: -1)
}
