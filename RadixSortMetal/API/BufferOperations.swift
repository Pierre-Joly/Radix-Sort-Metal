import Foundation
import Metal

extension MetalRadixSorter {
    // MARK: - Public API: Direct Command Buffer Execution

    public func sort(buffer: MTLBuffer, count: Int) throws {
        try validateSort(count: count, valuesBuffer: buffer, indexBuffer: nil)
        guard count > 1 else {
            return
        }

        let commandBuffer = try makeCommandBuffer()
        try encodeSort(buffer: buffer, count: count, into: commandBuffer)
        try commitAndWait(commandBuffer)
    }

    public func sort(buffer: MTLBuffer, indexBuffer: MTLBuffer, count: Int, initializeIndices: Bool = true) throws {
        try validateSort(count: count, valuesBuffer: buffer, indexBuffer: indexBuffer)
        let needsWork = count > 1 || (initializeIndices && count > 0)
        guard needsWork else {
            return
        }

        let commandBuffer = try makeCommandBuffer()
        try encodeSort(
            buffer: buffer,
            indexBuffer: indexBuffer,
            count: count,
            initializeIndices: initializeIndices,
            into: commandBuffer
        )
        try commitAndWait(commandBuffer)
    }

    public func reorder(buffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: buffer,
            destinationBuffer: buffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 1 else {
            return
        }

        let commandBuffer = try makeCommandBuffer()
        try encodeReorder(buffer: buffer, indexBuffer: indexBuffer, elementStride: elementStride, count: count, into: commandBuffer)
        try commitAndWait(commandBuffer)
    }

    public func reorder(
        buffer: MTLBuffer,
        temporaryBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: buffer,
            destinationBuffer: temporaryBuffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 1 else {
            return
        }

        let commandBuffer = try makeCommandBuffer()
        try encodeReorder(
            buffer: buffer,
            temporaryBuffer: temporaryBuffer,
            indexBuffer: indexBuffer,
            elementStride: elementStride,
            count: count,
            into: commandBuffer
        )
        try commitAndWait(commandBuffer)
    }

    public func reorder(
        sourceBuffer: MTLBuffer,
        destinationBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: sourceBuffer,
            destinationBuffer: destinationBuffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 0 else {
            return
        }

        let commandBuffer = try makeCommandBuffer()
        try encodeReorder(
            sourceBuffer: sourceBuffer,
            destinationBuffer: destinationBuffer,
            indexBuffer: indexBuffer,
            elementStride: elementStride,
            count: count,
            into: commandBuffer
        )
        try commitAndWait(commandBuffer)
    }

    // MARK: - Public API: Encode Into Existing MTLCommandBuffer

    public func encodeSort(buffer: MTLBuffer, count: Int, into commandBuffer: MTLCommandBuffer) throws {
        try validateSort(count: count, valuesBuffer: buffer, indexBuffer: nil)
        guard count > 1 else {
            return
        }

        try withOwnedComputeEncoder(on: commandBuffer, purpose: "sort compute encoder") { encoder in
            try encodeSort(buffer: buffer, count: count, using: encoder)
        }
    }

    public func encodeSort(
        buffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        count: Int,
        initializeIndices: Bool = true,
        into commandBuffer: MTLCommandBuffer
    ) throws {
        try validateSort(count: count, valuesBuffer: buffer, indexBuffer: indexBuffer)
        let needsWork = count > 1 || (initializeIndices && count > 0)
        guard needsWork else {
            return
        }

        try withOwnedComputeEncoder(on: commandBuffer, purpose: "sort compute encoder") { encoder in
            try encodeSort(
                buffer: buffer,
                indexBuffer: indexBuffer,
                count: count,
                initializeIndices: initializeIndices,
                using: encoder
            )
        }
    }

    public func encodeReorder(
        buffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int,
        into commandBuffer: MTLCommandBuffer
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: buffer,
            destinationBuffer: buffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 1 else {
            return
        }

        try withOwnedComputeEncoder(on: commandBuffer, purpose: "reorder compute encoder") { encoder in
            try encodeReorder(buffer: buffer, indexBuffer: indexBuffer, elementStride: elementStride, count: count, using: encoder)
        }
    }

    public func encodeReorder(
        buffer: MTLBuffer,
        temporaryBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int,
        into commandBuffer: MTLCommandBuffer
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: buffer,
            destinationBuffer: temporaryBuffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 1 else {
            return
        }

        try withOwnedComputeEncoder(on: commandBuffer, purpose: "reorder compute encoder") { encoder in
            try encodeReorder(
                buffer: buffer,
                temporaryBuffer: temporaryBuffer,
                indexBuffer: indexBuffer,
                elementStride: elementStride,
                count: count,
                using: encoder
            )
        }
    }

    public func encodeReorder(
        sourceBuffer: MTLBuffer,
        destinationBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int,
        into commandBuffer: MTLCommandBuffer
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: sourceBuffer,
            destinationBuffer: destinationBuffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 0 else {
            return
        }

        try withOwnedComputeEncoder(on: commandBuffer, purpose: "reorder compute encoder") { encoder in
            try encodeReorder(
                sourceBuffer: sourceBuffer,
                destinationBuffer: destinationBuffer,
                indexBuffer: indexBuffer,
                elementStride: elementStride,
                count: count,
                using: encoder
            )
        }
    }

    // MARK: - Public API: Encode Into Existing MTLComputeCommandEncoder

    public func encodeSort(buffer: MTLBuffer, count: Int, using encoder: MTLComputeCommandEncoder) throws {
        try validateSort(count: count, valuesBuffer: buffer, indexBuffer: nil)
        guard count > 1 else {
            return
        }

        try encodeSortInternal(valuesBuffer: buffer, indexBuffer: nil, count: count, initializeIndices: false, using: encoder)
    }

    public func encodeSort(
        buffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        count: Int,
        initializeIndices: Bool = true,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        try validateSort(count: count, valuesBuffer: buffer, indexBuffer: indexBuffer)
        let needsWork = count > 1 || (initializeIndices && count > 0)
        guard needsWork else {
            return
        }

        try encodeSortInternal(
            valuesBuffer: buffer,
            indexBuffer: indexBuffer,
            count: count,
            initializeIndices: initializeIndices,
            using: encoder
        )
    }

    public func encodeReorder(
        buffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: buffer,
            destinationBuffer: buffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 1 else {
            return
        }

        try encodeReorderInPlaceInternal(buffer: buffer, indexBuffer: indexBuffer, spec: spec, using: encoder)
    }

    public func encodeReorder(
        buffer: MTLBuffer,
        temporaryBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: buffer,
            destinationBuffer: temporaryBuffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 1 else {
            return
        }

        try encodeReorderInPlaceInternal(
            buffer: buffer,
            temporaryBuffer: temporaryBuffer,
            indexBuffer: indexBuffer,
            spec: spec,
            using: encoder
        )
    }

    public func encodeReorder(
        sourceBuffer: MTLBuffer,
        destinationBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        elementStride: Int,
        count: Int,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        let spec = try makeReorderSpec(
            count: count,
            elementStride: elementStride,
            sourceBuffer: sourceBuffer,
            destinationBuffer: destinationBuffer,
            indexBuffer: indexBuffer
        )
        guard spec.count > 0 else {
            return
        }

        try encodeReorderGather(
            sourceBuffer: sourceBuffer,
            destinationBuffer: destinationBuffer,
            indexBuffer: indexBuffer,
            spec: spec,
            using: encoder
        )
    }
}
