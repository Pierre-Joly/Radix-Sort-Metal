import Foundation
import Metal

extension MetalRadixSorter {
    // MARK: - Resource Allocation

    func allocateSortResources(count: Int, needsIndices: Bool) throws -> SortResources {
        let byteCount = try checkedByteCount(count: count, stride: MemoryLayout<UInt32>.stride)
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

    // MARK: - Validation and Specs

    func validateSort(count: Int, valuesBuffer: MTLBuffer, indexBuffer: MTLBuffer?) throws {
        guard count >= 0 else {
            throw SortError.invalidCount(count)
        }
        guard count <= Int(UInt32.max) else {
            throw SortError.unsupportedCount(count)
        }

        let requiredValueBytes = try checkedByteCount(count: count, stride: MemoryLayout<UInt32>.stride)
        if valuesBuffer.length < requiredValueBytes {
            throw SortError.bufferTooSmall(
                buffer: "values",
                requiredBytes: requiredValueBytes,
                availableBytes: valuesBuffer.length
            )
        }

        if let indexBuffer {
            let requiredIndexElements = count
            let availableIndexElements = indexBuffer.length / MemoryLayout<UInt32>.stride
            if requiredIndexElements > availableIndexElements {
                throw SortError.indexBufferTooSmall(
                    requiredElements: requiredIndexElements,
                    availableElements: availableIndexElements
                )
            }
        }
    }

    func makeReorderSpec(
        count: Int,
        elementStride: Int,
        sourceBuffer: MTLBuffer,
        destinationBuffer: MTLBuffer,
        indexBuffer: MTLBuffer
    ) throws -> ReorderSpec {
        guard count >= 0 else {
            throw SortError.invalidCount(count)
        }
        guard count <= Int(UInt32.max) else {
            throw SortError.unsupportedCount(count)
        }
        guard elementStride > 0 else {
            throw SortError.invalidElementStride(elementStride)
        }
        guard elementStride <= Int(UInt32.max) else {
            throw SortError.invalidElementStride(elementStride)
        }

        let requiredBytes = try checkedByteCount(count: count, stride: elementStride)
        guard requiredBytes <= Int(UInt32.max) else {
            throw SortError.unsupportedByteCount(requiredBytes)
        }

        if sourceBuffer.length < requiredBytes {
            throw SortError.bufferTooSmall(
                buffer: "source",
                requiredBytes: requiredBytes,
                availableBytes: sourceBuffer.length
            )
        }
        if destinationBuffer.length < requiredBytes {
            throw SortError.bufferTooSmall(
                buffer: "destination",
                requiredBytes: requiredBytes,
                availableBytes: destinationBuffer.length
            )
        }

        let requiredIndexElements = count
        let availableIndexElements = indexBuffer.length / MemoryLayout<UInt32>.stride
        if requiredIndexElements > availableIndexElements {
            throw SortError.indexBufferTooSmall(
                requiredElements: requiredIndexElements,
                availableElements: availableIndexElements
            )
        }

        return ReorderSpec(
            count: count,
            count32: UInt32(count),
            elementStride32: UInt32(elementStride),
            byteCount: requiredBytes,
            byteCount32: UInt32(requiredBytes)
        )
    }

    func checkedByteCount(count: Int, stride: Int) throws -> Int {
        let (byteCount, overflow) = count.multipliedReportingOverflow(by: stride)
        if overflow {
            throw SortError.unsupportedByteCount(Int.max)
        }
        return byteCount
    }

    // MARK: - Buffer Helpers

    func makeSharedUIntBuffer(_ values: [UInt32], label: String) throws -> MTLBuffer {
        let byteCount = try checkedByteCount(count: values.count, stride: MemoryLayout<UInt32>.stride)
        guard let buffer = device.makeBuffer(bytes: values, length: byteCount, options: .storageModeShared) else {
            throw SortError.bufferAllocationFailed(label)
        }
        return buffer
    }

    func makeSharedUIntBuffer(count: Int, label: String) throws -> MTLBuffer {
        let byteCount = try checkedByteCount(count: count, stride: MemoryLayout<UInt32>.stride)
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw SortError.bufferAllocationFailed(label)
        }
        return buffer
    }

    func makeSharedBuffer<T>(from values: [T], byteCount: Int, label: String) throws -> MTLBuffer {
        let buffer = values.withUnsafeBytes { rawBytes -> MTLBuffer? in
            guard let baseAddress = rawBytes.baseAddress else {
                return nil
            }
            return device.makeBuffer(bytes: baseAddress, length: byteCount, options: .storageModeShared)
        }

        guard let buffer else {
            throw SortError.bufferAllocationFailed(label)
        }
        return buffer
    }

    func readArray<T>(from buffer: MTLBuffer, count: Int, as _: T.Type) -> [T] {
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    // MARK: - Command Helpers

    func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw SortError.commandBufferCreationFailed
        }
        return commandBuffer
    }

    func withOwnedComputeEncoder(
        on commandBuffer: MTLCommandBuffer,
        purpose: String,
        _ body: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SortError.commandEncodingFailed(purpose)
        }
        defer {
            encoder.endEncoding()
        }
        try body(encoder)
    }

    func commitAndWait(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw SortError.commandBufferFailed(commandBuffer.error)
        }
    }

    // MARK: - Shared Compute Steps

    func encodeInitializeIndices(indexBuffer: MTLBuffer, count: UInt32, using encoder: MTLComputeCommandEncoder) throws {
        var countValue = count
        encoder.setComputePipelineState(initializeIndicesPipeline)
        encoder.setBuffer(indexBuffer, offset: 0, index: 0)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 1)
        dispatch1D(pipeline: initializeIndicesPipeline, elementCount: Int(count), using: encoder)
    }

    func encodeCopyUInt(source: MTLBuffer, destination: MTLBuffer, count: UInt32, using encoder: MTLComputeCommandEncoder) throws {
        var countValue = count
        encoder.setComputePipelineState(copyUIntBufferPipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)
        dispatch1D(pipeline: copyUIntBufferPipeline, elementCount: Int(count), using: encoder)
    }

    func encodeCopyBytes(source: MTLBuffer, destination: MTLBuffer, byteCount: UInt32, using encoder: MTLComputeCommandEncoder) throws {
        var byteCountValue = byteCount
        encoder.setComputePipelineState(copyByteBufferPipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        encoder.setBytes(&byteCountValue, length: MemoryLayout<UInt32>.stride, index: 2)
        dispatch1D(pipeline: copyByteBufferPipeline, elementCount: Int(byteCount), using: encoder)
    }

    func dispatchFixedThreadgroups(groupCount: Int, using encoder: MTLComputeCommandEncoder) {
        guard groupCount > 0 else {
            return
        }
        let threadgroup = MTLSize(width: Constants.threadsPerThreadgroup, height: 1, depth: 1)
        let grid = MTLSize(width: groupCount, height: 1, depth: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
    }

    func dispatch1D(
        pipeline: MTLComputePipelineState,
        elementCount: Int,
        using encoder: MTLComputeCommandEncoder
    ) {
        guard elementCount > 0 else {
            return
        }
        let width = max(1, min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup))
        let grid = MTLSize(width: elementCount, height: 1, depth: 1)
        let threadgroup = MTLSize(width: width, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
    }

    // MARK: - Pipeline Loading

    static func makePipeline(function: String, library: MTLLibrary, device: MTLDevice) throws -> MTLComputePipelineState {
        guard let kernel = library.makeFunction(name: function) else {
            throw SortError.kernelFunctionNotFound(function)
        }
        do {
            return try device.makeComputePipelineState(function: kernel)
        } catch {
            throw SortError.pipelineCreationFailed(function: function, error: error)
        }
    }

    static func validateThreads(for pipeline: MTLComputePipelineState, function: String) throws {
        if pipeline.maxTotalThreadsPerThreadgroup < Constants.threadsPerThreadgroup {
            throw SortError.unsupportedThreadgroupSize(
                function: function,
                required: Constants.threadsPerThreadgroup,
                maxSupported: pipeline.maxTotalThreadsPerThreadgroup
            )
        }
    }

    static func loadKernelLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: .module) {
            return library
        }

        // Fallback for environments where SwiftPM resources provide .metal sources
        // but no precompiled default metallib.
        let kernelSource = try loadKernelSource()
        do {
            return try device.makeLibrary(source: kernelSource, options: nil)
        } catch {
            throw SortError.kernelCompilationFailed(error)
        }
    }

    static func loadKernelSource() throws -> String {
        guard let resourceRoot = Bundle.module.resourceURL else {
            throw SortError.kernelSourceMissing
        }
        let fileManager = FileManager.default
        let searchDirectories = [
            resourceRoot.appendingPathComponent("Kernels", isDirectory: true),
            resourceRoot
        ]
        let preferredKernelOrder = [
            "Common.metal",
            "UtilityKernels.metal",
            "HistogramKernels.metal",
            "ScatterKernels.metal"
        ]
        let preferredKernelIndex = Dictionary(
            uniqueKeysWithValues: preferredKernelOrder.enumerated().map { index, name in (name, index) }
        )

        var metalFiles: [URL] = []
        for directory in searchDirectories {
            guard let urls = try? fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            let candidates = urls
                .filter { $0.pathExtension == "metal" && $0.lastPathComponent != "RadixSortMetal.metal" }
            guard !candidates.isEmpty else {
                continue
            }

            let canonicalCandidates = candidates.filter { preferredKernelIndex[$0.lastPathComponent] != nil }
            let selectedCandidates = canonicalCandidates.isEmpty ? candidates : canonicalCandidates

            metalFiles = selectedCandidates.sorted { lhs, rhs in
                let lhsName = lhs.lastPathComponent
                let rhsName = rhs.lastPathComponent
                let lhsRank = preferredKernelIndex[lhsName] ?? Int.max
                let rhsRank = preferredKernelIndex[rhsName] ?? Int.max

                if lhsRank != rhsRank {
                    return lhsRank < rhsRank
                }
                return lhsName < rhsName
            }
            break
        }

        guard !metalFiles.isEmpty else {
            throw SortError.kernelSourceMissing
        }

        var sourceChunks: [String] = []
        sourceChunks.reserveCapacity(metalFiles.count)

        for fileURL in metalFiles {
            do {
                sourceChunks.append(try String(contentsOf: fileURL, encoding: .utf8))
            } catch {
                throw SortError.kernelSourceUnreadable(error)
            }
        }

        return sourceChunks.joined(separator: "\n\n")
    }
}
