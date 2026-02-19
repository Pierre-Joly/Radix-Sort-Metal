import Foundation
import Metal

extension MetalRadixSorter {
    // MARK: - Sort Encoding Internals

    func encodeSortInternal(
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

        let resources = try allocateSortResources(count: count, needsIndices: indexBuffer != nil)
        var state = SortState(
            inputValues: valuesBuffer,
            outputValues: resources.scratchValues,
            inputIndices: indexBuffer,
            outputIndices: resources.scratchIndices
        )

        for pass in 0..<Constants.passes {
            encodeSortPass(pass: pass, count32: count32, resources: resources, state: &state, using: encoder)
        }

        try copySortResultsBackIfNeeded(
            finalState: state,
            valuesBuffer: valuesBuffer,
            indexBuffer: indexBuffer,
            count32: count32,
            using: encoder
        )
    }

    func encodeSortPass(
        pass: Int,
        count32: UInt32,
        resources: SortResources,
        state: inout SortState,
        using encoder: MTLComputeCommandEncoder
    ) {
        let shift = UInt32(pass * Constants.radixBits)

        encodeClearTotalHistogram(totalHistogram: resources.totalHistogram, using: encoder)

        encodeCountBlockHistograms(
            inputValues: state.inputValues,
            blockHistograms: resources.blockHistograms,
            totalHistogram: resources.totalHistogram,
            count32: count32,
            shift: shift,
            blockCount: resources.blockCount,
            using: encoder
        )

        encodeScanTotalHistogram(totalHistogram: resources.totalHistogram, binOffsets: resources.binOffsets, using: encoder)

        encodeBuildBlockOffsets(
            blockHistograms: resources.blockHistograms,
            binOffsets: resources.binOffsets,
            blockOffsets: resources.blockOffsets,
            blockCount: resources.blockCount,
            using: encoder
        )

        if let inputIndices = state.inputIndices, let outputIndices = state.outputIndices {
            encodeScatterValuesAndIndices(
                inputValues: state.inputValues,
                outputValues: state.outputValues,
                inputIndices: inputIndices,
                outputIndices: outputIndices,
                blockOffsets: resources.blockOffsets,
                count32: count32,
                shift: shift,
                blockCount: resources.blockCount,
                using: encoder
            )
        } else {
            encodeScatterValuesOnly(
                inputValues: state.inputValues,
                outputValues: state.outputValues,
                blockOffsets: resources.blockOffsets,
                count32: count32,
                shift: shift,
                blockCount: resources.blockCount,
                using: encoder
            )
        }

        state.advanceToNextPass()
    }

    func encodeClearTotalHistogram(totalHistogram: MTLBuffer, using encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(clearHistogramPipeline)
        encoder.setBuffer(totalHistogram, offset: 0, index: 0)
        dispatchFixedThreadgroups(groupCount: 1, using: encoder)
    }

    func encodeCountBlockHistograms(
        inputValues: MTLBuffer,
        blockHistograms: MTLBuffer,
        totalHistogram: MTLBuffer,
        count32: UInt32,
        shift: UInt32,
        blockCount: Int,
        using encoder: MTLComputeCommandEncoder
    ) {
        var countValue = count32
        var shiftValue = shift

        encoder.setComputePipelineState(countHistogramPipeline)
        encoder.setBuffer(inputValues, offset: 0, index: 0)
        encoder.setBuffer(blockHistograms, offset: 0, index: 1)
        encoder.setBuffer(totalHistogram, offset: 0, index: 2)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&shiftValue, length: MemoryLayout<UInt32>.stride, index: 4)
        dispatchFixedThreadgroups(groupCount: blockCount, using: encoder)
    }

    func encodeScanTotalHistogram(
        totalHistogram: MTLBuffer,
        binOffsets: MTLBuffer,
        using encoder: MTLComputeCommandEncoder
    ) {
        encoder.setComputePipelineState(scanHistogramPipeline)
        encoder.setBuffer(totalHistogram, offset: 0, index: 0)
        encoder.setBuffer(binOffsets, offset: 0, index: 1)
        dispatchFixedThreadgroups(groupCount: 1, using: encoder)
    }

    func encodeBuildBlockOffsets(
        blockHistograms: MTLBuffer,
        binOffsets: MTLBuffer,
        blockOffsets: MTLBuffer,
        blockCount: Int,
        using encoder: MTLComputeCommandEncoder
    ) {
        var blockCountValue = UInt32(blockCount)

        encoder.setComputePipelineState(buildBlockOffsetsPipeline)
        encoder.setBuffer(blockHistograms, offset: 0, index: 0)
        encoder.setBuffer(binOffsets, offset: 0, index: 1)
        encoder.setBuffer(blockOffsets, offset: 0, index: 2)
        encoder.setBytes(&blockCountValue, length: MemoryLayout<UInt32>.stride, index: 3)
        dispatchFixedThreadgroups(groupCount: 1, using: encoder)
    }

    func encodeScatterValuesOnly(
        inputValues: MTLBuffer,
        outputValues: MTLBuffer,
        blockOffsets: MTLBuffer,
        count32: UInt32,
        shift: UInt32,
        blockCount: Int,
        using encoder: MTLComputeCommandEncoder
    ) {
        var countValue = count32
        var shiftValue = shift

        encoder.setComputePipelineState(scatterValuesPipeline)
        encoder.setBuffer(inputValues, offset: 0, index: 0)
        encoder.setBuffer(outputValues, offset: 0, index: 1)
        encoder.setBuffer(blockOffsets, offset: 0, index: 2)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&shiftValue, length: MemoryLayout<UInt32>.stride, index: 4)
        dispatchFixedThreadgroups(groupCount: blockCount, using: encoder)
    }

    func encodeScatterValuesAndIndices(
        inputValues: MTLBuffer,
        outputValues: MTLBuffer,
        inputIndices: MTLBuffer,
        outputIndices: MTLBuffer,
        blockOffsets: MTLBuffer,
        count32: UInt32,
        shift: UInt32,
        blockCount: Int,
        using encoder: MTLComputeCommandEncoder
    ) {
        var countValue = count32
        var shiftValue = shift

        encoder.setComputePipelineState(scatterKeyIndexPipeline)
        encoder.setBuffer(inputValues, offset: 0, index: 0)
        encoder.setBuffer(outputValues, offset: 0, index: 1)
        encoder.setBuffer(inputIndices, offset: 0, index: 2)
        encoder.setBuffer(outputIndices, offset: 0, index: 3)
        encoder.setBuffer(blockOffsets, offset: 0, index: 4)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&shiftValue, length: MemoryLayout<UInt32>.stride, index: 6)
        dispatchFixedThreadgroups(groupCount: blockCount, using: encoder)
    }

    func copySortResultsBackIfNeeded(
        finalState: SortState,
        valuesBuffer: MTLBuffer,
        indexBuffer: MTLBuffer?,
        count32: UInt32,
        using encoder: MTLComputeCommandEncoder
    ) throws {
        guard Constants.passes % 2 != 0 else {
            return
        }

        try encodeCopyUInt(source: finalState.inputValues, destination: valuesBuffer, count: count32, using: encoder)
        if let indexBuffer, let inputIndices = finalState.inputIndices {
            try encodeCopyUInt(source: inputIndices, destination: indexBuffer, count: count32, using: encoder)
        }
    }
}
