import Foundation
import Metal

public struct IndexedSortResult {
    public let values: [UInt32]
    public let indices: [UInt32]

    public init(values: [UInt32], indices: [UInt32]) {
        self.values = values
        self.indices = indices
    }
}

public final class MetalRadixSorter {
    public enum SortError: Error {
        case metalUnavailable
        case commandQueueCreationFailed
        case kernelSourceMissing
        case kernelSourceUnreadable(Error)
        case kernelFunctionNotFound(String)
        case kernelCompilationFailed(Error)
        case pipelineCreationFailed(function: String, error: Error)
        case unsupportedThreadgroupSize(function: String, required: Int, maxSupported: Int)
        case invalidCount(Int)
        case indexBufferTooSmall(requiredElements: Int, availableElements: Int)
        case unsupportedCount(Int)
        case bufferAllocationFailed(String)
        case commandBufferCreationFailed
        case commandEncodingFailed(String)
        case commandBufferFailed(Error?)
    }

    private enum Constants {
        static let radixBits = 8
        static let radix = 1 << radixBits
        static let passes = 32 / radixBits
        static let elementsPerBlock = 2_048
        static let threadsPerThreadgroup = 256
    }

    private struct SortResources {
        let scratchValues: MTLBuffer
        let scratchIndices: MTLBuffer?
        let blockHistograms: MTLBuffer
        let totalHistogram: MTLBuffer
        let binOffsets: MTLBuffer
        let blockOffsets: MTLBuffer
        let blockCount: Int
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let initializeIndicesPipeline: MTLComputePipelineState
    private let clearHistogramPipeline: MTLComputePipelineState
    private let copyBufferPipeline: MTLComputePipelineState
    private let countHistogramPipeline: MTLComputePipelineState
    private let scanHistogramPipeline: MTLComputePipelineState
    private let buildBlockOffsetsPipeline: MTLComputePipelineState
    private let scatterValuesPipeline: MTLComputePipelineState
    private let scatterKeyIndexPipeline: MTLComputePipelineState

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw SortError.metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw SortError.commandQueueCreationFailed
        }

        let kernelSource = try Self.loadKernelSource()
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource, options: nil)
        } catch {
            throw SortError.kernelCompilationFailed(error)
        }

        self.device = device
        self.commandQueue = commandQueue
        self.initializeIndicesPipeline = try Self.makePipeline(function: "initialize_indices", library: library, device: device)
        self.clearHistogramPipeline = try Self.makePipeline(function: "clear_histogram_256", library: library, device: device)
        self.copyBufferPipeline = try Self.makePipeline(function: "copy_uint_buffer", library: library, device: device)
        self.countHistogramPipeline = try Self.makePipeline(function: "count_block_histograms", library: library, device: device)
        self.scanHistogramPipeline = try Self.makePipeline(function: "scan_total_histogram", library: library, device: device)
        self.buildBlockOffsetsPipeline = try Self.makePipeline(function: "build_block_offsets", library: library, device: device)
        self.scatterValuesPipeline = try Self.makePipeline(function: "scatter_values_stable", library: library, device: device)
        self.scatterKeyIndexPipeline = try Self.makePipeline(function: "scatter_key_index_stable", library: library, device: device)

        try Self.validateThreads(for: clearHistogramPipeline, function: "clear_histogram_256")
        try Self.validateThreads(for: countHistogramPipeline, function: "count_block_histograms")
        try Self.validateThreads(for: scanHistogramPipeline, function: "scan_total_histogram")
        try Self.validateThreads(for: buildBlockOffsetsPipeline, function: "build_block_offsets")
        try Self.validateThreads(for: scatterValuesPipeline, function: "scatter_values_stable")
        try Self.validateThreads(for: scatterKeyIndexPipeline, function: "scatter_key_index_stable")
    }

    public func sort(_ values: [UInt32]) throws -> [UInt32] {
        guard !values.isEmpty else {
            return []
        }

        let byteCount = values.count * MemoryLayout<UInt32>.stride
        guard let buffer = device.makeBuffer(bytes: values, length: byteCount, options: .storageModeShared) else {
            throw SortError.bufferAllocationFailed("input buffer")
        }

        try sort(buffer: buffer, count: values.count)

        let pointer = buffer.contents().bindMemory(to: UInt32.self, capacity: values.count)
        return Array(UnsafeBufferPointer(start: pointer, count: values.count))
    }

    public func sortInPlace(_ values: inout [UInt32]) throws {
        values = try sort(values)
    }

    public func sortWithIndices(_ values: [UInt32]) throws -> IndexedSortResult {
        guard !values.isEmpty else {
            return IndexedSortResult(values: [], indices: [])
        }

        let byteCount = values.count * MemoryLayout<UInt32>.stride
        guard let valuesBuffer = device.makeBuffer(bytes: values, length: byteCount, options: .storageModeShared) else {
            throw SortError.bufferAllocationFailed("indexed input buffer")
        }
        guard let indexBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw SortError.bufferAllocationFailed("indexed output buffer")
        }

        try sort(buffer: valuesBuffer, indexBuffer: indexBuffer, count: values.count, initializeIndices: true)

        let valuesPointer = valuesBuffer.contents().bindMemory(to: UInt32.self, capacity: values.count)
        let indicesPointer = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: values.count)
        let sortedValues = Array(UnsafeBufferPointer(start: valuesPointer, count: values.count))
        let sortedIndices = Array(UnsafeBufferPointer(start: indicesPointer, count: values.count))
        return IndexedSortResult(values: sortedValues, indices: sortedIndices)
    }

    public func sort(buffer: MTLBuffer, count: Int) throws {
        try validate(count: count, valuesBuffer: buffer, indexBuffer: nil)
        guard count > 1 else {
            return
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw SortError.commandBufferCreationFailed
        }

        try encodeSort(buffer: buffer, count: count, into: commandBuffer)
        try commitAndWait(commandBuffer)
    }

    public func sort(buffer: MTLBuffer, indexBuffer: MTLBuffer, count: Int, initializeIndices: Bool = true) throws {
        try validate(count: count, valuesBuffer: buffer, indexBuffer: indexBuffer)
        let needsWork = count > 1 || (initializeIndices && count > 0)
        guard needsWork else {
            return
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw SortError.commandBufferCreationFailed
        }

        try encodeSort(buffer: buffer, indexBuffer: indexBuffer, count: count, initializeIndices: initializeIndices, into: commandBuffer)
        try commitAndWait(commandBuffer)
    }

    public func encodeSort(buffer: MTLBuffer, count: Int, into commandBuffer: MTLCommandBuffer) throws {
        try validate(count: count, valuesBuffer: buffer, indexBuffer: nil)
        guard count > 1 else {
            return
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SortError.commandEncodingFailed("sort compute encoder")
        }
        defer {
            encoder.endEncoding()
        }
        try encodeSort(buffer: buffer, count: count, using: encoder)
    }

    public func encodeSort(
        buffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        count: Int,
        initializeIndices: Bool = true,
        into commandBuffer: MTLCommandBuffer
    ) throws {
        try validate(count: count, valuesBuffer: buffer, indexBuffer: indexBuffer)
        let needsWork = count > 1 || (initializeIndices && count > 0)
        guard needsWork else {
            return
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SortError.commandEncodingFailed("sort compute encoder")
        }
        defer {
            encoder.endEncoding()
        }
        try encodeSort(
            buffer: buffer,
            indexBuffer: indexBuffer,
            count: count,
            initializeIndices: initializeIndices,
            using: encoder
        )
    }

    public func encodeSort(buffer: MTLBuffer, count: Int, using encoder: MTLComputeCommandEncoder) throws {
        try validate(count: count, valuesBuffer: buffer, indexBuffer: nil)
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
        try validate(count: count, valuesBuffer: buffer, indexBuffer: indexBuffer)
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

    private func encodeSortInternal(
        valuesBuffer: MTLBuffer,
        indexBuffer: MTLBuffer?,
        count: Int,
        initializeIndices: Bool,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        let count32 = UInt32(count)
        if initializeIndices, let indexBuffer, count > 0 {
            try encodeInitializeIndices(indexBuffer: indexBuffer, count: count32, using: encoder)
        }

        guard count > 1 else {
            return
        }

        let resources = try allocateResources(count: count, needsIndices: indexBuffer != nil)
        var inputValues = valuesBuffer
        var outputValues = resources.scratchValues
        var inputIndices = indexBuffer
        var outputIndices = resources.scratchIndices
        var blockCount32 = UInt32(resources.blockCount)

        let threadgroupSize = MTLSize(width: Constants.threadsPerThreadgroup, height: 1, depth: 1)
        let perBlockGrid = MTLSize(width: resources.blockCount, height: 1, depth: 1)
        let singleGrid = MTLSize(width: 1, height: 1, depth: 1)

        for pass in 0..<Constants.passes {
            var countValue = count32
            var shift = UInt32(pass * Constants.radixBits)

            encoder.setComputePipelineState(clearHistogramPipeline)
            encoder.setBuffer(resources.totalHistogram, offset: 0, index: 0)
            encoder.dispatchThreadgroups(singleGrid, threadsPerThreadgroup: threadgroupSize)

            encoder.setComputePipelineState(countHistogramPipeline)
            encoder.setBuffer(inputValues, offset: 0, index: 0)
            encoder.setBuffer(resources.blockHistograms, offset: 0, index: 1)
            encoder.setBuffer(resources.totalHistogram, offset: 0, index: 2)
            encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&shift, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(perBlockGrid, threadsPerThreadgroup: threadgroupSize)

            encoder.setComputePipelineState(scanHistogramPipeline)
            encoder.setBuffer(resources.totalHistogram, offset: 0, index: 0)
            encoder.setBuffer(resources.binOffsets, offset: 0, index: 1)
            encoder.dispatchThreadgroups(singleGrid, threadsPerThreadgroup: threadgroupSize)

            encoder.setComputePipelineState(buildBlockOffsetsPipeline)
            encoder.setBuffer(resources.blockHistograms, offset: 0, index: 0)
            encoder.setBuffer(resources.binOffsets, offset: 0, index: 1)
            encoder.setBuffer(resources.blockOffsets, offset: 0, index: 2)
            encoder.setBytes(&blockCount32, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(singleGrid, threadsPerThreadgroup: threadgroupSize)

            if let inputIndices, let outputIndices {
                encoder.setComputePipelineState(scatterKeyIndexPipeline)
                encoder.setBuffer(inputValues, offset: 0, index: 0)
                encoder.setBuffer(outputValues, offset: 0, index: 1)
                encoder.setBuffer(inputIndices, offset: 0, index: 2)
                encoder.setBuffer(outputIndices, offset: 0, index: 3)
                encoder.setBuffer(resources.blockOffsets, offset: 0, index: 4)
                encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 5)
                encoder.setBytes(&shift, length: MemoryLayout<UInt32>.stride, index: 6)
            } else {
                encoder.setComputePipelineState(scatterValuesPipeline)
                encoder.setBuffer(inputValues, offset: 0, index: 0)
                encoder.setBuffer(outputValues, offset: 0, index: 1)
                encoder.setBuffer(resources.blockOffsets, offset: 0, index: 2)
                encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
                encoder.setBytes(&shift, length: MemoryLayout<UInt32>.stride, index: 4)
            }
            encoder.dispatchThreadgroups(perBlockGrid, threadsPerThreadgroup: threadgroupSize)

            swap(&inputValues, &outputValues)
            if inputIndices != nil {
                swap(&inputIndices, &outputIndices)
            }
        }

        if Constants.passes % 2 != 0 {
            try encodeCopy(source: inputValues, destination: valuesBuffer, count: count32, using: encoder)
            if let indexBuffer, let inputIndices {
                try encodeCopy(source: inputIndices, destination: indexBuffer, count: count32, using: encoder)
            }
        }
    }

    private func allocateResources(count: Int, needsIndices: Bool) throws -> SortResources {
        let byteCount = count * MemoryLayout<UInt32>.stride
        let blockCount = (count + Constants.elementsPerBlock - 1) / Constants.elementsPerBlock
        let radixBytes = Constants.radix * MemoryLayout<UInt32>.stride
        let blockTableBytes = blockCount * radixBytes

        guard let scratchValues = device.makeBuffer(length: byteCount, options: .storageModePrivate) else {
            throw SortError.bufferAllocationFailed("scratch values buffer")
        }
        let scratchIndices: MTLBuffer?
        if needsIndices {
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModePrivate) else {
                throw SortError.bufferAllocationFailed("scratch indices buffer")
            }
            scratchIndices = buffer
        } else {
            scratchIndices = nil
        }

        guard let blockHistograms = device.makeBuffer(length: blockTableBytes, options: .storageModePrivate) else {
            throw SortError.bufferAllocationFailed("block histogram buffer")
        }
        guard let totalHistogram = device.makeBuffer(length: radixBytes, options: .storageModePrivate) else {
            throw SortError.bufferAllocationFailed("total histogram buffer")
        }
        guard let binOffsets = device.makeBuffer(length: radixBytes, options: .storageModePrivate) else {
            throw SortError.bufferAllocationFailed("bin offsets buffer")
        }
        guard let blockOffsets = device.makeBuffer(length: blockTableBytes, options: .storageModePrivate) else {
            throw SortError.bufferAllocationFailed("block offsets buffer")
        }

        return SortResources(
            scratchValues: scratchValues,
            scratchIndices: scratchIndices,
            blockHistograms: blockHistograms,
            totalHistogram: totalHistogram,
            binOffsets: binOffsets,
            blockOffsets: blockOffsets,
            blockCount: blockCount
        )
    }

    private func commitAndWait(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw SortError.commandBufferFailed(commandBuffer.error)
        }
    }

    private func validate(count: Int, valuesBuffer: MTLBuffer, indexBuffer: MTLBuffer?) throws {
        guard count >= 0 else {
            throw SortError.invalidCount(count)
        }
        guard count <= valuesBuffer.length / MemoryLayout<UInt32>.stride else {
            throw SortError.invalidCount(count)
        }
        guard count <= Int(UInt32.max) else {
            throw SortError.unsupportedCount(count)
        }
        if let indexBuffer {
            let availableIndexCount = indexBuffer.length / MemoryLayout<UInt32>.stride
            if count > availableIndexCount {
                throw SortError.indexBufferTooSmall(requiredElements: count, availableElements: availableIndexCount)
            }
        }
    }

    private func encodeInitializeIndices(indexBuffer: MTLBuffer, count: UInt32, using encoder: MTLComputeCommandEncoder) throws {
        var countValue = count
        encoder.setComputePipelineState(initializeIndicesPipeline)
        encoder.setBuffer(indexBuffer, offset: 0, index: 0)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 1)

        let width = max(1, min(initializeIndicesPipeline.threadExecutionWidth, initializeIndicesPipeline.maxTotalThreadsPerThreadgroup))
        let grid = MTLSize(width: Int(count), height: 1, depth: 1)
        let threadgroup = MTLSize(width: width, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
    }

    private func encodeCopy(source: MTLBuffer, destination: MTLBuffer, count: UInt32, using encoder: MTLComputeCommandEncoder) throws {
        var countValue = count
        encoder.setComputePipelineState(copyBufferPipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)

        let width = max(1, min(copyBufferPipeline.threadExecutionWidth, copyBufferPipeline.maxTotalThreadsPerThreadgroup))
        let grid = MTLSize(width: Int(count), height: 1, depth: 1)
        let threadgroup = MTLSize(width: width, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
    }

    private static func makePipeline(function: String, library: MTLLibrary, device: MTLDevice) throws -> MTLComputePipelineState {
        guard let kernel = library.makeFunction(name: function) else {
            throw SortError.kernelFunctionNotFound(function)
        }
        do {
            return try device.makeComputePipelineState(function: kernel)
        } catch {
            throw SortError.pipelineCreationFailed(function: function, error: error)
        }
    }

    private static func validateThreads(for pipeline: MTLComputePipelineState, function: String) throws {
        if pipeline.maxTotalThreadsPerThreadgroup < Constants.threadsPerThreadgroup {
            throw SortError.unsupportedThreadgroupSize(
                function: function,
                required: Constants.threadsPerThreadgroup,
                maxSupported: pipeline.maxTotalThreadsPerThreadgroup
            )
        }
    }

    private static func loadKernelSource() throws -> String {
        guard let url = Bundle.module.url(forResource: "RadixSortMetal", withExtension: "metal") else {
            throw SortError.kernelSourceMissing
        }
        do {
            return try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw SortError.kernelSourceUnreadable(error)
        }
    }
}
