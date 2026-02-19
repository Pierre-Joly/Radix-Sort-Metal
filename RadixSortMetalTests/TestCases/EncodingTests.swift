import Metal
import XCTest
@testable import RadixSortMetal

extension MetalRadixSorterTests {
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
        let payload: [UInt32] = [80, 20, 60, 21, 90, 10]
        let payloadB: [Payload] = [
            Payload(id: 0, value: 800),
            Payload(id: 1, value: 200),
            Payload(id: 2, value: 600),
            Payload(id: 3, value: 210),
            Payload(id: 4, value: 900),
            Payload(id: 5, value: 100)
        ]
        let byteCount = keys.count * MemoryLayout<UInt32>.stride
        let payloadBByteCount = payloadB.count * MemoryLayout<Payload>.stride
        guard let keyBuffer = device.makeBuffer(bytes: keys, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate key buffer.")
            return
        }
        guard let indexBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }
        guard let payloadBuffer = device.makeBuffer(bytes: payload, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate payload buffer.")
            return
        }
        guard let payloadBBuffer = device.makeBuffer(bytes: payloadB, length: payloadBByteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate second payload buffer.")
            return
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer.")
            return
        }

        try sorter.encodeSort(buffer: keyBuffer, indexBuffer: indexBuffer, count: keys.count, into: commandBuffer)
        try sorter.encodeReorder(
            buffer: payloadBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<UInt32>.stride,
            count: payload.count,
            into: commandBuffer
        )
        try sorter.encodeReorder(
            buffer: payloadBBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<Payload>.stride,
            count: payloadB.count,
            into: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, .completed)

        let keyPtr = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let idxPtr = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let sortedKeys = Array(UnsafeBufferPointer(start: keyPtr, count: keys.count))
        let permutation = Array(UnsafeBufferPointer(start: idxPtr, count: keys.count))

        XCTAssertEqual(sortedKeys, [1, 2, 2, 6, 8, 9])
        XCTAssertEqual(permutation.map { keys[Int($0)] }, sortedKeys)

        let payloadPtr = payloadBuffer.contents().bindMemory(to: UInt32.self, capacity: payload.count)
        let sortedPayload = Array(UnsafeBufferPointer(start: payloadPtr, count: payload.count))
        XCTAssertEqual(sortedPayload, [10, 20, 21, 60, 80, 90])

        let payloadBPtr = payloadBBuffer.contents().bindMemory(to: Payload.self, capacity: payloadB.count)
        let sortedPayloadB = Array(UnsafeBufferPointer(start: payloadBPtr, count: payloadB.count))
        XCTAssertEqual(sortedPayloadB.map(\.id), [5, 1, 3, 2, 0, 4])
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

        let keys: [UInt32] = [13, 4, 7, 2, 11, 1]
        let payload: [UInt32] = [130, 40, 70, 20, 110, 10]
        let payloadB: [UInt32] = [13, 4, 7, 2, 11, 1]
        let byteCount = keys.count * MemoryLayout<UInt32>.stride
        guard let keyBuffer = device.makeBuffer(bytes: keys, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate key buffer.")
            return
        }
        guard let indexBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }
        guard let payloadBuffer = device.makeBuffer(bytes: payload, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate payload buffer.")
            return
        }
        guard let payloadBBuffer = device.makeBuffer(bytes: payloadB, length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate second payload buffer.")
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

        try sorter.encodeSort(buffer: keyBuffer, indexBuffer: indexBuffer, count: keys.count, using: encoder)
        try sorter.encodeReorder(
            buffer: payloadBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<UInt32>.stride,
            count: payload.count,
            using: encoder
        )
        try sorter.encodeReorder(
            buffer: payloadBBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<UInt32>.stride,
            count: payloadB.count,
            using: encoder
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, .completed)

        let keyPtr = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let output = Array(UnsafeBufferPointer(start: keyPtr, count: keys.count))
        XCTAssertEqual(output, keys.sorted())

        let payloadPtr = payloadBuffer.contents().bindMemory(to: UInt32.self, capacity: payload.count)
        let sortedPayload = Array(UnsafeBufferPointer(start: payloadPtr, count: payload.count))
        XCTAssertEqual(sortedPayload, [10, 20, 40, 70, 110, 130])

        let payloadBPtr = payloadBBuffer.contents().bindMemory(to: UInt32.self, capacity: payloadB.count)
        let sortedPayloadB = Array(UnsafeBufferPointer(start: payloadBPtr, count: payloadB.count))
        XCTAssertEqual(sortedPayloadB, [1, 2, 4, 7, 11, 13])
    }
}
