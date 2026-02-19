import Metal
import XCTest
@testable import RadixSortMetal

final class MetalRadixSorterTests: XCTestCase {
    func testSortsFixedInput() throws {
        let sorter = try makeSorter()
        let input: [UInt32] = [42, 17, 3, 999, 17, 0, 128, 128, 4_294_967_295]
        let output = try sorter.sort(input)
        XCTAssertEqual(output, input.sorted())
    }

    func testSortsRandomInputsAcrossSizes() throws {
        let sorter = try makeSorter()
        var generator = LCG(seed: 0x1234_5678_9ABC_DEF0)
        let sizes = [0, 1, 2, 7, 33, 255, 256, 2_049, 8_192, 65_537]

        for size in sizes {
            var values: [UInt32] = []
            values.reserveCapacity(size)
            for _ in 0..<size {
                values.append(generator.next())
            }

            let sorted = try sorter.sort(values)
            XCTAssertEqual(sorted, values.sorted(), "Mismatch for size \(size)")
        }
    }

    func testInPlaceSortArray() throws {
        let sorter = try makeSorter()
        var input: [UInt32] = [11, 7, 3, 2, 19, 5, 1, 13]
        try sorter.sortInPlace(&input)
        XCTAssertEqual(input, [1, 2, 3, 5, 7, 11, 13, 19])
    }

    func testSortsBufferInPlace() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }

        let input: [UInt32] = [301, 22, 77, 22, 1, 5000, 99]
        let byteCount = input.count * MemoryLayout<UInt32>.stride
        guard let buffer = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate test buffer.")
            return
        }

        try sorter.sort(buffer: buffer, count: input.count)

        let pointer = buffer.contents().bindMemory(to: UInt32.self, capacity: input.count)
        let output = Array(UnsafeBufferPointer(start: pointer, count: input.count))
        XCTAssertEqual(output, input.sorted())
    }

    func testSortWithIndicesReturnsPermutation() throws {
        let sorter = try makeSorter()
        let keys: [UInt32] = [50, 10, 50, 1, 10]
        let payload: [UInt32] = [500, 100, 501, 10, 101]

        let result = try sorter.sortWithIndices(keys)
        XCTAssertEqual(result.values, [1, 10, 10, 50, 50])

        let reorderedPayload = result.indices.map { payload[Int($0)] }
        XCTAssertEqual(reorderedPayload, [10, 100, 101, 500, 501])
    }

    func testSortBufferWithIndicesInPlace() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }

        let keys: [UInt32] = [9, 3, 12, 3, 1]
        let byteCount = keys.count * MemoryLayout<UInt32>.stride
        guard let keyBuffer = device.makeBuffer(bytes: keys, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate key buffer.")
            return
        }
        guard let indexBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }

        try sorter.sort(buffer: keyBuffer, indexBuffer: indexBuffer, count: keys.count)

        let keyPtr = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let idxPtr = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let sortedKeys = Array(UnsafeBufferPointer(start: keyPtr, count: keys.count))
        let permutation = Array(UnsafeBufferPointer(start: idxPtr, count: keys.count))

        XCTAssertEqual(sortedKeys, [1, 3, 3, 9, 12])
        XCTAssertEqual(permutation.map { keys[Int($0)] }, sortedKeys)
    }

    func testEncodeSortIntoExistingCommandBuffer() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }
        guard let commandQueue = device.makeCommandQueue() else {
            XCTFail("Failed to create command queue.")
            return
        }

        let keys: [UInt32] = [8, 2, 6, 2, 9, 1]
        let byteCount = keys.count * MemoryLayout<UInt32>.stride
        guard let keyBuffer = device.makeBuffer(bytes: keys, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate key buffer.")
            return
        }
        guard let indexBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer.")
            return
        }

        try sorter.encodeSort(buffer: keyBuffer, indexBuffer: indexBuffer, count: keys.count, into: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, .completed)

        let keyPtr = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let idxPtr = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let sortedKeys = Array(UnsafeBufferPointer(start: keyPtr, count: keys.count))
        let permutation = Array(UnsafeBufferPointer(start: idxPtr, count: keys.count))

        XCTAssertEqual(sortedKeys, [1, 2, 2, 6, 8, 9])
        XCTAssertEqual(permutation.map { keys[Int($0)] }, sortedKeys)
    }

    func testEncodeSortUsingExistingComputeEncoder() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }
        guard let commandQueue = device.makeCommandQueue() else {
            XCTFail("Failed to create command queue.")
            return
        }

        let values: [UInt32] = [13, 4, 7, 2, 11, 1]
        let byteCount = values.count * MemoryLayout<UInt32>.stride
        guard let valueBuffer = device.makeBuffer(bytes: values, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate values buffer.")
            return
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer.")
            return
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Failed to create compute encoder.")
            return
        }

        try sorter.encodeSort(buffer: valueBuffer, count: values.count, using: encoder)
        try sorter.encodeSort(buffer: valueBuffer, count: values.count, using: encoder)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, .completed)

        let pointer = valueBuffer.contents().bindMemory(to: UInt32.self, capacity: values.count)
        let output = Array(UnsafeBufferPointer(start: pointer, count: values.count))
        XCTAssertEqual(output, values.sorted())
    }

    private func makeSorter() throws -> MetalRadixSorter {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is not available on this host.")
        }
        return try MetalRadixSorter()
    }
}

private struct LCG {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt32 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return UInt32(truncatingIfNeeded: state >> 16)
    }
}
