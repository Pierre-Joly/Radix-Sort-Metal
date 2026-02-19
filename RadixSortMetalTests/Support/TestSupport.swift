import Metal
import XCTest
@testable import RadixSortMetal

extension MetalRadixSorterTests {
    func makeSorter() throws -> MetalRadixSorter {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available on this host.")
        }
        return try MetalRadixSorter()
    }
}

struct Payload: Equatable {
    var id: UInt32
    var value: UInt32
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
