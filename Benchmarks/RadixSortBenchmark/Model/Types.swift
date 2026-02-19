import Foundation

struct BenchmarkRow {
    let size: Int
    let repetitions: Int
    let cpuMedianMs: Double
    let cpuMeanMs: Double
    let gpuMedianMs: Double
    let gpuMeanMs: Double

    var speedup: Double {
        cpuMedianMs / gpuMedianMs
    }
}

enum BenchmarkError: Error {
    case missingOutputArgument
    case inconsistentSort(size: Int, repetition: Int)
}

struct LCG {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt32 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return UInt32(truncatingIfNeeded: state >> 16)
    }
}
