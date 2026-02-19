import Foundation
import Metal

extension MetalRadixSorter {
    // MARK: - Reorder Encoding Internals

    func encodeReorderInPlaceInternal(
        buffer: MTLBuffer,
        temporaryBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        spec: ReorderSpec,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        guard spec.count > 1 else {
            return
        }
        guard temporaryBuffer !== buffer else {
            throw SortError.aliasingBuffersNotSupported(
                operation: "reorder in-place temporary buffer must differ from source buffer"
            )
        }

        try encodeReorderGather(
            sourceBuffer: buffer,
            destinationBuffer: temporaryBuffer,
            indexBuffer: indexBuffer,
            spec: spec,
            using: encoder
        )
        try encodeCopyBytes(source: temporaryBuffer, destination: buffer, byteCount: spec.byteCount32, using: encoder)
    }

    func encodeReorderInPlaceInternal(
        buffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        spec: ReorderSpec,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        guard spec.count > 1 else {
            return
        }
        guard let scratch = device.makeBuffer(length: spec.byteCount, options: .storageModePrivate) else {
            throw SortError.bufferAllocationFailed("reorder scratch buffer")
        }
        try encodeReorderInPlaceInternal(
            buffer: buffer,
            temporaryBuffer: scratch,
            indexBuffer: indexBuffer,
            spec: spec,
            using: encoder
        )
    }

    func encodeReorderGather(
        sourceBuffer: MTLBuffer,
        destinationBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        spec: ReorderSpec,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        var countValue = spec.count32
        var strideValue = spec.elementStride32

        encoder.setComputePipelineState(reorderByIndexPipeline)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
        encoder.setBuffer(destinationBuffer, offset: 0, index: 1)
        encoder.setBuffer(indexBuffer, offset: 0, index: 2)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&strideValue, length: MemoryLayout<UInt32>.stride, index: 4)
        dispatch1D(pipeline: reorderByIndexPipeline, elementCount: spec.count, using: encoder)
    }
}
