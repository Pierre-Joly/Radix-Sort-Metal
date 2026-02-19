import Metal
import XCTest
@testable import RadixSortMetal

extension MetalRadixSorterTests {
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

    func testSortBufferWithIndicesAndPayloadReorder() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }

        let keys: [UInt32] = [9, 3, 12, 3, 1]
        let payload: [Payload] = [
            Payload(id: 0, value: 900),
            Payload(id: 1, value: 300),
            Payload(id: 2, value: 1200),
            Payload(id: 3, value: 301),
            Payload(id: 4, value: 100)
        ]
        let payloadB: [UInt32] = [90, 30, 120, 31, 10]

        let keyBytes = keys.count * MemoryLayout<UInt32>.stride
        let payloadBytes = payload.count * MemoryLayout<Payload>.stride
        let payloadBBytes = payloadB.count * MemoryLayout<UInt32>.stride
        guard let keyBuffer = device.makeBuffer(bytes: keys, length: keyBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate key buffer.")
            return
        }
        guard let indexBuffer = device.makeBuffer(length: keyBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }
        guard let payloadBuffer = device.makeBuffer(bytes: payload, length: payloadBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate payload buffer.")
            return
        }
        guard let payloadBBuffer = device.makeBuffer(bytes: payloadB, length: payloadBBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate second payload buffer.")
            return
        }

        try sorter.sort(buffer: keyBuffer, indexBuffer: indexBuffer, count: keys.count)
        try sorter.reorder(
            buffer: payloadBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<Payload>.stride,
            count: payload.count
        )
        try sorter.reorder(
            buffer: payloadBBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<UInt32>.stride,
            count: payloadB.count
        )

        let keyPtr = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let sortedKeys = Array(UnsafeBufferPointer(start: keyPtr, count: keys.count))
        XCTAssertEqual(sortedKeys, [1, 3, 3, 9, 12])

        let payloadPtr = payloadBuffer.contents().bindMemory(to: Payload.self, capacity: payload.count)
        let sortedPayload = Array(UnsafeBufferPointer(start: payloadPtr, count: payload.count))
        XCTAssertEqual(sortedPayload.map(\.id), [4, 1, 3, 0, 2])

        let payloadBPtr = payloadBBuffer.contents().bindMemory(to: UInt32.self, capacity: payloadB.count)
        let sortedPayloadB = Array(UnsafeBufferPointer(start: payloadBPtr, count: payloadB.count))
        XCTAssertEqual(sortedPayloadB, [10, 30, 31, 90, 120])
    }

    func testReorderSourceToDestinationUsingPermutation() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }

        let indices: [UInt32] = [2, 0, 1]
        let source: [Payload] = [
            Payload(id: 10, value: 1000),
            Payload(id: 11, value: 1100),
            Payload(id: 12, value: 1200)
        ]
        let destination = [Payload](repeating: Payload(id: 0, value: 0), count: source.count)

        let indexBytes = indices.count * MemoryLayout<UInt32>.stride
        let payloadBytes = source.count * MemoryLayout<Payload>.stride

        guard let indexBuffer = device.makeBuffer(bytes: indices, length: indexBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }
        guard let sourceBuffer = device.makeBuffer(bytes: source, length: payloadBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate source buffer.")
            return
        }
        guard let destinationBuffer = device.makeBuffer(bytes: destination, length: payloadBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate destination buffer.")
            return
        }

        try sorter.reorder(
            sourceBuffer: sourceBuffer,
            destinationBuffer: destinationBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<Payload>.stride,
            count: source.count
        )

        let outPtr = destinationBuffer.contents().bindMemory(to: Payload.self, capacity: source.count)
        let reordered = Array(UnsafeBufferPointer(start: outPtr, count: source.count))
        XCTAssertEqual(reordered.map(\.id), [12, 10, 11])
    }

    func testReorderInPlaceUsingProvidedTemporaryBuffer() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }

        let indices: [UInt32] = [2, 0, 1]
        let values: [Payload] = [
            Payload(id: 10, value: 1000),
            Payload(id: 11, value: 1100),
            Payload(id: 12, value: 1200)
        ]
        let zeroed = [Payload](repeating: Payload(id: 0, value: 0), count: values.count)

        let indexBytes = indices.count * MemoryLayout<UInt32>.stride
        let payloadBytes = values.count * MemoryLayout<Payload>.stride

        guard let indexBuffer = device.makeBuffer(bytes: indices, length: indexBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }
        guard let valueBuffer = device.makeBuffer(bytes: values, length: payloadBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate value buffer.")
            return
        }
        guard let temporaryBuffer = device.makeBuffer(bytes: zeroed, length: payloadBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate temporary buffer.")
            return
        }

        try sorter.reorder(
            buffer: valueBuffer,
            temporaryBuffer: temporaryBuffer,
            indexBuffer: indexBuffer,
            elementStride: MemoryLayout<Payload>.stride,
            count: values.count
        )

        let outPtr = valueBuffer.contents().bindMemory(to: Payload.self, capacity: values.count)
        let reordered = Array(UnsafeBufferPointer(start: outPtr, count: values.count))
        XCTAssertEqual(reordered.map(\.id), [12, 10, 11])
    }

    func testReorderWithProvidedTemporaryBufferRejectsAliasing() throws {
        let sorter = try makeSorter()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this host.")
        }

        let indices: [UInt32] = [0, 1, 2]
        let values: [Payload] = [
            Payload(id: 10, value: 1000),
            Payload(id: 11, value: 1100),
            Payload(id: 12, value: 1200)
        ]
        let indexBytes = indices.count * MemoryLayout<UInt32>.stride
        let payloadBytes = values.count * MemoryLayout<Payload>.stride

        guard let indexBuffer = device.makeBuffer(bytes: indices, length: indexBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate index buffer.")
            return
        }
        guard let valueBuffer = device.makeBuffer(bytes: values, length: payloadBytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate value buffer.")
            return
        }

        do {
            try sorter.reorder(
                buffer: valueBuffer,
                temporaryBuffer: valueBuffer,
                indexBuffer: indexBuffer,
                elementStride: MemoryLayout<Payload>.stride,
                count: values.count
            )
            XCTFail("Expected aliasingBuffersNotSupported error.")
        } catch let MetalRadixSorter.SortError.aliasingBuffersNotSupported(operation) {
            XCTAssertFalse(operation.isEmpty)
        }
    }
}
