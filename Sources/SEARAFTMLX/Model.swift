//
//  Model.swift
//  SEARAFTMLX
//
//  SEA-RAFT — isomorphic to sea-raft-mlx's model.py (itself isomorphic to
//  princeton-vl/SEA-RAFT @ 9137517). NHWC throughout. Weight keys match the
//  mlx-community/SEA-RAFT-*-mlx checkpoints 1:1.
//

import Foundation
import MLX
import MLXNN

public struct SEARAFTConfig: Sendable {
    public var dim: Int = 128
    public var initialDim: Int = 64
    public var blockDims: [Int] = [64, 128, 256]
    public var radius: Int = 4
    public var numBlocks: Int = 2
    public var iters: Int = 4

    public var corrLevels: Int { 4 }
    public var corrChannel: Int { corrLevels * (radius * 2 + 1) * (radius * 2 + 1) }

    public init() {}

    /// The S variants (resnet18 backbone) — both published mlx-community checkpoints.
    public static let s = SEARAFTConfig()
}

// MARK: - layer.py

final class ConvNextBlock: Module, UnaryLayer {
    @ModuleInfo var dwconv: Conv2d
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    @ParameterInfo var gamma: MLXArray
    @ModuleInfo var final: Conv2d

    init(_ dim: Int, _ outputDim: Int) {
        self._dwconv.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim,
                                           kernelSize: 7, padding: 3, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, 4 * outputDim)
        self._pwconv2.wrappedValue = Linear(4 * outputDim, dim)
        self._gamma.wrappedValue = 1e-6 * MLXArray.ones([dim])
        self._final.wrappedValue = Conv2d(inputChannels: dim, outputChannels: outputDim,
                                          kernelSize: 1, padding: 0)
    }

    func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = dwconv(xIn)
        x = norm(x)
        x = pwconv1(x)
        x = gelu(x)
        x = pwconv2(x)
        x = gamma * x
        return final(xIn + x)
    }
}

final class BasicBlock: Module, UnaryLayer {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var bn1: BatchNorm
    @ModuleInfo var bn2: BatchNorm
    @ModuleInfo var bn3: BatchNorm?
    @ModuleInfo var downsample: [Conv2d]?

    init(_ inPlanes: Int, _ planes: Int, stride: Int = 1) {
        self._conv1.wrappedValue = Conv2d(inputChannels: inPlanes, outputChannels: planes,
                                          kernelSize: 3, stride: IntOrPair(stride), padding: 1)
        self._conv2.wrappedValue = Conv2d(inputChannels: planes, outputChannels: planes,
                                          kernelSize: 3, stride: 1, padding: 1)
        self._bn1.wrappedValue = BatchNorm(featureCount: planes)
        self._bn2.wrappedValue = BatchNorm(featureCount: planes)
        if stride == 1 && inPlanes == planes {
            self._bn3.wrappedValue = nil
            self._downsample.wrappedValue = nil
        } else {
            self._bn3.wrappedValue = BatchNorm(featureCount: planes)
            self._downsample.wrappedValue = [Conv2d(inputChannels: inPlanes, outputChannels: planes,
                                                    kernelSize: 1, stride: IntOrPair(stride))]
        }
    }

    func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var y = relu(bn1(conv1(xIn)))
        y = relu(bn2(conv2(y)))
        var x = xIn
        if let downsample, let bn3 {
            x = bn3(downsample[0](x))
        }
        return relu(x + y)
    }
}

// MARK: - extractor.py

final class ResNetFPN: Module {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var bn1: BatchNorm
    @ModuleInfo var layer1: [BasicBlock]
    @ModuleInfo var layer2: [BasicBlock]
    @ModuleInfo var layer3: [BasicBlock]
    @ModuleInfo(key: "final_conv") var finalConv: Conv2d

    init(_ config: SEARAFTConfig, inputDim: Int, outputDim: Int) {
        let dims = config.blockDims
        self._conv1.wrappedValue = Conv2d(inputChannels: inputDim, outputChannels: config.initialDim,
                                          kernelSize: 7, stride: 2, padding: 3)
        self._bn1.wrappedValue = BatchNorm(featureCount: config.initialDim)
        var inPlanes = config.initialDim

        func makeLayer(_ dim: Int, stride: Int, num: Int) -> [BasicBlock] {
            var layers = [BasicBlock(inPlanes, dim, stride: stride)]
            for _ in 0..<(num - 1) {
                layers.append(BasicBlock(dim, dim, stride: 1))
            }
            inPlanes = dim
            return layers
        }

        self._layer1.wrappedValue = makeLayer(dims[0], stride: 1, num: 2)
        self._layer2.wrappedValue = makeLayer(dims[1], stride: 2, num: 2)
        self._layer3.wrappedValue = makeLayer(dims[2], stride: 2, num: 2)
        self._finalConv.wrappedValue = Conv2d(inputChannels: dims[2], outputChannels: outputDim,
                                              kernelSize: 1, stride: 1)
    }

    func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = relu(bn1(conv1(xIn)))
        for blk in layer1 { x = blk(x) }
        for blk in layer2 { x = blk(x) }
        for blk in layer3 { x = blk(x) }
        return finalConv(x)
    }
}

// MARK: - update.py

final class BasicMotionEncoder: Module {
    @ModuleInfo var convc1: Conv2d
    @ModuleInfo var convc2: Conv2d
    @ModuleInfo var convf1: Conv2d
    @ModuleInfo var convf2: Conv2d
    @ModuleInfo var conv: Conv2d

    init(_ config: SEARAFTConfig, dim: Int) {
        let corPlanes = config.corrChannel
        self._convc1.wrappedValue = Conv2d(inputChannels: corPlanes, outputChannels: dim * 2,
                                           kernelSize: 1, padding: 0)
        self._convc2.wrappedValue = Conv2d(inputChannels: dim * 2, outputChannels: dim + dim / 2,
                                           kernelSize: 3, padding: 1)
        self._convf1.wrappedValue = Conv2d(inputChannels: 2, outputChannels: dim,
                                           kernelSize: 7, padding: 3)
        self._convf2.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim / 2,
                                           kernelSize: 3, padding: 1)
        self._conv.wrappedValue = Conv2d(inputChannels: dim * 2, outputChannels: dim - 2,
                                         kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ flow: MLXArray, _ corr: MLXArray) -> MLXArray {
        var cor = relu(convc1(corr))
        cor = relu(convc2(cor))
        var flo = relu(convf1(flow))
        flo = relu(convf2(flo))
        let out = relu(conv(concatenated([cor, flo], axis: -1)))
        return concatenated([out, flow], axis: -1)
    }
}

final class BasicUpdateBlock: Module {
    @ModuleInfo var encoder: BasicMotionEncoder
    @ModuleInfo var refine: [ConvNextBlock]

    init(_ config: SEARAFTConfig, hdim: Int, cdim: Int) {
        self._encoder.wrappedValue = BasicMotionEncoder(config, dim: cdim)
        self._refine.wrappedValue = (0..<config.numBlocks).map { _ in
            ConvNextBlock(2 * cdim + hdim, hdim)
        }
    }

    func callAsFunction(_ netIn: MLXArray, _ inpIn: MLXArray, _ corr: MLXArray, _ flow: MLXArray) -> MLXArray {
        let motionFeatures = encoder(flow, corr)
        let inp = concatenated([inpIn, motionFeatures], axis: -1)
        var net = netIn
        for blk in refine {
            net = blk(concatenated([net, inp], axis: -1))
        }
        return net
    }
}

// MARK: - corr.py

func coordsGrid(batch: Int, ht: Int, wd: Int) -> MLXArray {
    // [N, H, W, 2] (x, y) pixel coords
    let xs = broadcast(MLXArray(Array(0..<wd).map { Float($0) }).reshaped([1, 1, wd, 1]),
                       to: [batch, ht, wd, 1])
    let ys = broadcast(MLXArray(Array(0..<ht).map { Float($0) }).reshaped([1, ht, 1, 1]),
                       to: [batch, ht, wd, 1])
    return concatenated([xs, ys], axis: -1)
}

final class CorrBlock {
    let numLevels: Int
    let radius: Int
    var corrPyramid: [MLXArray] = []
    let delta: MLXArray   // [1, 2r+1, 2r+1, 2] — torch's (dy, dx) enumeration, matched exactly

    init(fmap1: MLXArray, fmap2: MLXArray, config: SEARAFTConfig) {
        self.numLevels = config.corrLevels
        self.radius = config.radius
        let N = fmap1.shape[0], h1 = fmap1.shape[1], w1 = fmap1.shape[2], d = fmap1.shape[3]
        let a = fmap1.reshaped([N, h1 * w1, d])
        let scale = 1.0 / Float(d).squareRoot()
        var f2 = fmap2
        for level in 0..<numLevels {
            let h2 = f2.shape[1], w2 = f2.shape[2]
            let b = f2.reshaped([N, h2 * w2, d])
            var corr = matmul(a, b.transposed(0, 2, 1)) * scale     // [N, h1w1, h2w2]
            corr = corr.reshaped([N * h1 * w1, h2, w2, 1])
            corrPyramid.append(corr)
            if level < numLevels - 1 {
                f2 = interpolateBilinear(f2, scaleFactor: 0.5, alignCorners: false)
            }
        }
        // delta: stack(meshgrid(dy, dx)) -> last dim (dy[i], dx[j]); added to (x, y) coords —
        // the reference's trained-in enumeration; match it, don't "fix" it.
        let r = Float(radius)
        let side = 2 * radius + 1
        let lin = MLXArray(Array(0..<side).map { Float($0) - r })
        let dyg = broadcast(lin.reshaped([side, 1]), to: [side, side])
        let dxg = broadcast(lin.reshaped([1, side]), to: [side, side])
        self.delta = stacked([dyg, dxg], axis: -1).reshaped([1, side, side, 2])
    }

    func callAsFunction(_ coords: MLXArray) -> MLXArray {
        let N = coords.shape[0], h1 = coords.shape[1], w1 = coords.shape[2]
        let centroidBase = coords.reshaped([N * h1 * w1, 1, 1, 2])
        var outPyramid = [MLXArray]()
        for i in 0..<numLevels {
            let corr = corrPyramid[i]
            let h2 = corr.shape[1], w2 = corr.shape[2]
            let coordsLvl = centroidBase / Float(pow(2.0, Double(i))) + delta   // pixel (x, y)
            let gx = 2 * coordsLvl[0..., 0..., 0..., 0] / Float(w2 - 1) - 1
            let gy = 2 * coordsLvl[0..., 0..., 0..., 1] / Float(h2 - 1) - 1
            let grid = stacked([gx, gy], axis: -1)
            let sampled = gridSampleBilinearZeros(corr, grid: grid, alignCorners: true)
            outPyramid.append(sampled.reshaped([N, h1, w1, -1]))
        }
        return concatenated(outPyramid, axis: -1)
    }
}

// MARK: - raft.py

public final class SEARAFT: Module {
    public let config: SEARAFTConfig

    @ModuleInfo var cnet: ResNetFPN
    @ModuleInfo(key: "init_conv") var initConv: Conv2d
    @ModuleInfo(key: "upsample_weight") var upsampleWeight: [Conv2d]
    @ModuleInfo(key: "flow_head") var flowHead: [Conv2d]
    @ModuleInfo var fnet: ResNetFPN
    @ModuleInfo(key: "update_block") var updateBlock: BasicUpdateBlock

    public init(_ config: SEARAFTConfig = .s) {
        self.config = config
        let c = config
        self._cnet.wrappedValue = ResNetFPN(c, inputDim: 6, outputDim: 2 * c.dim)
        self._initConv.wrappedValue = Conv2d(inputChannels: 2 * c.dim, outputChannels: 2 * c.dim,
                                             kernelSize: 3, padding: 1)
        self._upsampleWeight.wrappedValue = [
            Conv2d(inputChannels: c.dim, outputChannels: c.dim * 2, kernelSize: 3, padding: 1),
            Conv2d(inputChannels: c.dim * 2, outputChannels: 64 * 9, kernelSize: 1, padding: 0),
        ]
        self._flowHead.wrappedValue = [
            Conv2d(inputChannels: c.dim, outputChannels: 2 * c.dim, kernelSize: 3, padding: 1),
            Conv2d(inputChannels: 2 * c.dim, outputChannels: 6, kernelSize: 3, padding: 1),
        ]
        self._fnet.wrappedValue = ResNetFPN(c, inputDim: 3, outputDim: 2 * c.dim)
        self._updateBlock.wrappedValue = BasicUpdateBlock(c, hdim: c.dim, cdim: c.dim)
    }

    private func flowHeadOut(_ net: MLXArray) -> MLXArray {
        flowHead[1](relu(flowHead[0](net)))
    }

    private func upsampleWeightOut(_ net: MLXArray) -> MLXArray {
        upsampleWeight[1](relu(upsampleWeight[0](net)))
    }

    /// Convex-combination 8× upsample (mask channels k*64 + i*8 + j; softmax over k).
    func upsampleData(flow: MLXArray, info: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        let N = flow.shape[0], H = flow.shape[1], W = flow.shape[2]
        var m = mask.reshaped([N, H, W, 9, 8, 8])
        m = softmax(m, axis: 3)

        func convex(_ x: MLXArray) -> MLXArray {
            let C = x.shape[3]
            let patches = unfold3x3(x)                          // [N,H,W,C,9]
            // out[n,h,w,i,j,c] = sum_k patches[n,h,w,c,k] * m[n,h,w,k,i,j]
            // einsum via reshape-matmul: [N*H*W, (8*8), 9] @ [N*H*W, 9, C]
            let pm = patches.reshaped([N * H * W, C, 9]).transposed(0, 2, 1)   // [B', 9, C]
            let mm = m.reshaped([N * H * W, 9, 64]).transposed(0, 2, 1)        // [B', 64, 9]
            let out = matmul(mm, pm)                                            // [B', 64, C]
            // [N,H,W,8,8,C] -> [N,H,8,W,8,C] -> [N,8H,8W,C]
            return out.reshaped([N, H, W, 8, 8, C]).transposed(0, 1, 3, 2, 4, 5)
                .reshaped([N, H * 8, W * 8, C])
        }

        return (convex(8 * flow), convex(info))
    }

    /// image1/2: `[N, H, W, 3]` RGB in 0…255. Returns the final flow `[N, H, W, 2]` (pixels).
    public func callAsFunction(_ image1In: MLXArray, _ image2In: MLXArray, iters: Int? = nil) -> MLXArray {
        let c = config
        let steps = iters ?? c.iters
        let H0 = image1In.shape[1], W0 = image1In.shape[2]

        var image1 = 2 * (image1In / 255.0) - 1.0
        var image2 = 2 * (image2In / 255.0) - 1.0

        // InputPadder (sintel mode, /8, replicate)
        let padH = (((H0 / 8) + 1) * 8 - H0) % 8
        let padW = (((W0 / 8) + 1) * 8 - W0) % 8
        let pads = (padW / 2, padW - padW / 2, padH / 2, padH - padH / 2)
        if padH > 0 || padW > 0 {
            let widths = [IntOrPair((0, 0)), IntOrPair((pads.2, pads.3)),
                          IntOrPair((pads.0, pads.1)), IntOrPair((0, 0))]
            image1 = padded(image1, widths: widths, mode: .edge)
            image2 = padded(image2, widths: widths, mode: .edge)
        }
        let H = image1.shape[1], W = image1.shape[2]
        let N = image1.shape[0]

        var cnetOut = cnet(concatenated([image1, image2], axis: -1))
        cnetOut = initConv(cnetOut)
        var net = cnetOut[0..., 0..., 0..., ..<c.dim]
        let context = cnetOut[0..., 0..., 0..., c.dim...]

        var flowUpdate = flowHeadOut(net)
        var weightUpdate = 0.25 * upsampleWeightOut(net)
        var flow8 = flowUpdate[0..., 0..., 0..., ..<2]
        var info8 = flowUpdate[0..., 0..., 0..., 2...]
        var (flowUp, _) = upsampleData(flow: flow8, info: info8, mask: weightUpdate)

        let fmap1 = fnet(image1)
        let fmap2 = fnet(image2)
        let corrFn = CorrBlock(fmap1: fmap1, fmap2: fmap2, config: c)

        let base = coordsGrid(batch: N, ht: H / 8, wd: W / 8)
        for _ in 0..<steps {
            let coords2 = base + flow8
            let corr = corrFn(coords2)
            net = updateBlock(net, context, corr, flow8)
            flowUpdate = flowHeadOut(net)
            weightUpdate = 0.25 * upsampleWeightOut(net)
            flow8 = flow8 + flowUpdate[0..., 0..., 0..., ..<2]
            info8 = flowUpdate[0..., 0..., 0..., 2...]
            (flowUp, _) = upsampleData(flow: flow8, info: info8, mask: weightUpdate)
        }

        if padH > 0 || padW > 0 {
            return flowUp[0..., pads.2..<(H - pads.3), pads.0..<(W - pads.1), 0...]
        }
        return flowUp
    }
}

// MARK: - Weight loading

public enum SEARAFTError: Error, CustomStringConvertible {
    case weightsFileNotFound(String)
    case loadFailed(String)

    public var description: String {
        switch self {
        case .weightsFileNotFound(let p): return "SEA-RAFT weights not found: \(p)"
        case .loadFailed(let d): return "SEA-RAFT weight load failed: \(d)"
        }
    }
}

extension SEARAFT {
    /// Load an mlx-community SEA-RAFT checkpoint (`model.safetensors`, keys 1:1).
    public func loadWeights(from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SEARAFTError.weightsFileNotFound(url.path)
        }
        do {
            let arrays = try MLX.loadArrays(url: url)
            let parameters = ModuleParameters.unflattened(arrays)
            try update(parameters: parameters, verify: .noUnusedKeys)
            MLX.eval(self.parameters())
        } catch let e as SEARAFTError {
            throw e
        } catch {
            throw SEARAFTError.loadFailed(error.localizedDescription)
        }
    }
}
