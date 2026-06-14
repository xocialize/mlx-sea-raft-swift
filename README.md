# mlx-sea-raft-swift

The MLXEngine **`opticalFlow`** package over [SEA-RAFT](https://github.com/xocialize/sea-raft-mlx-swift) (Wang/Lipson/Deng, ECCV 2024, BSD-3) — dense per-pixel motion on Apple Silicon, the **temporal building block** of the visual optimization tier.

Estimates a `FlowField` (interleaved (u,v) pixel displacements) between two frames — for
backward-warping in temporal-consistent enhancement, and motion features for the pipeline
planner. The core is **parity-locked** (cosine 0.99997 / EPE 0.116 px vs the Python reference,
itself cosine-1.0 locked vs PyTorch). The `iters` knob trades quality for speed.

## Variants

| Variant | Checkpoint | Notes |
|---|---|---|
| `.spring` (default) | `mlx-community/SEA-RAFT-Tartan-C-T-TSKH-spring540x960-S-mlx` | full schedule, best accuracy |
| `.tartan` | `mlx-community/SEA-RAFT-Tartan480x640-S-mlx` | TartanAir/CC-BY-only provenance chain |

Checkpoint license **confirmed BSD-3 by the authors** in [princeton-vl/SEA-RAFT#31](https://github.com/princeton-vl/SEA-RAFT/issues/31#issuecomment-4674222973) (linked from the weight cards).

## Usage

```swift
import MLXServeCore
import MLXSEARAFT

let engine = MLXServeEngine()
try await engine.register(SEARAFTOpticalFlowPackage.registration, configuration: SEARAFTConfiguration())

let resp = try await engine.run(OpticalFlowRequest(image0: frameA, image1: frameB)) as! OpticalFlowResponse
let (u, v) = resp.flow[x, y]   // pixel displacement at (x, y)
```

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-sea-raft-swift", from: "0.1.0")`, then import `MLXSEARAFT` (the conformant `opticalFlow` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port MIT; weights BSD-3 (Princeton VL Lab).
