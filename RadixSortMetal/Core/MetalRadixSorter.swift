import Foundation
import Metal

public final class MetalRadixSorter {
    // MARK: - Errors

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
        case mismatchedElementCount(expected: Int, got: Int)
        case invalidElementStride(Int)
        case indexBufferTooSmall(requiredElements: Int, availableElements: Int)
        case bufferTooSmall(buffer: String, requiredBytes: Int, availableBytes: Int)
        case unsupportedCount(Int)
        case unsupportedByteCount(Int)
        case bufferAllocationFailed(String)
        case commandBufferCreationFailed
        case commandEncodingFailed(String)
        case commandBufferFailed(Error?)
        case aliasingBuffersNotSupported(operation: String)
    }

    // MARK: - Constants and Internal Models

    enum Constants {
        static let radixBits = 8
        static let radix = 1 << radixBits
        static let passes = 32 / radixBits
        static let elementsPerBlock = 2_048
        static let threadsPerThreadgroup = 256
    }

    enum Kernel {
        static let initializeIndices = "initialize_indices"
        static let clearHistogram = "clear_histogram_256"
        static let copyUIntBuffer = "copy_uint_buffer"
        static let copyByteBuffer = "copy_byte_buffer"
        static let reorderByIndex = "reorder_by_index_bytes"
        static let countHistogram = "count_block_histograms"
        static let scanHistogram = "scan_total_histogram"
        static let buildBlockOffsets = "build_block_offsets"
        static let scatterValues = "scatter_values_stable"
        static let scatterKeyIndex = "scatter_key_index_stable"
    }

    struct SortResources {
        let scratchValues: MTLBuffer
        let scratchIndices: MTLBuffer?
        let blockHistograms: MTLBuffer
        let totalHistogram: MTLBuffer
        let binOffsets: MTLBuffer
        let blockOffsets: MTLBuffer
        let blockCount: Int
    }

    struct SortState {
        var inputValues: MTLBuffer
        var outputValues: MTLBuffer
        var inputIndices: MTLBuffer?
        var outputIndices: MTLBuffer?

        mutating func advanceToNextPass() {
            swap(&inputValues, &outputValues)
            if inputIndices != nil {
                swap(&inputIndices, &outputIndices)
            }
        }
    }

    struct ReorderSpec {
        let count: Int
        let count32: UInt32
        let elementStride32: UInt32
        let byteCount: Int
        let byteCount32: UInt32
    }

    // MARK: - Metal State

    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    let initializeIndicesPipeline: MTLComputePipelineState
    let clearHistogramPipeline: MTLComputePipelineState
    let copyUIntBufferPipeline: MTLComputePipelineState
    let copyByteBufferPipeline: MTLComputePipelineState
    let reorderByIndexPipeline: MTLComputePipelineState
    let countHistogramPipeline: MTLComputePipelineState
    let scanHistogramPipeline: MTLComputePipelineState
    let buildBlockOffsetsPipeline: MTLComputePipelineState
    let scatterValuesPipeline: MTLComputePipelineState
    let scatterKeyIndexPipeline: MTLComputePipelineState

    // MARK: - Initialization

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw SortError.metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw SortError.commandQueueCreationFailed
        }

        let library = try Self.loadKernelLibrary(device: device)

        self.device = device
        self.commandQueue = commandQueue
        self.initializeIndicesPipeline = try Self.makePipeline(function: Kernel.initializeIndices, library: library, device: device)
        self.clearHistogramPipeline = try Self.makePipeline(function: Kernel.clearHistogram, library: library, device: device)
        self.copyUIntBufferPipeline = try Self.makePipeline(function: Kernel.copyUIntBuffer, library: library, device: device)
        self.copyByteBufferPipeline = try Self.makePipeline(function: Kernel.copyByteBuffer, library: library, device: device)
        self.reorderByIndexPipeline = try Self.makePipeline(function: Kernel.reorderByIndex, library: library, device: device)
        self.countHistogramPipeline = try Self.makePipeline(function: Kernel.countHistogram, library: library, device: device)
        self.scanHistogramPipeline = try Self.makePipeline(function: Kernel.scanHistogram, library: library, device: device)
        self.buildBlockOffsetsPipeline = try Self.makePipeline(function: Kernel.buildBlockOffsets, library: library, device: device)
        self.scatterValuesPipeline = try Self.makePipeline(function: Kernel.scatterValues, library: library, device: device)
        self.scatterKeyIndexPipeline = try Self.makePipeline(function: Kernel.scatterKeyIndex, library: library, device: device)

        try Self.validateThreads(for: clearHistogramPipeline, function: Kernel.clearHistogram)
        try Self.validateThreads(for: countHistogramPipeline, function: Kernel.countHistogram)
        try Self.validateThreads(for: scanHistogramPipeline, function: Kernel.scanHistogram)
        try Self.validateThreads(for: buildBlockOffsetsPipeline, function: Kernel.buildBlockOffsets)
        try Self.validateThreads(for: scatterValuesPipeline, function: Kernel.scatterValues)
        try Self.validateThreads(for: scatterKeyIndexPipeline, function: Kernel.scatterKeyIndex)
    }
}
