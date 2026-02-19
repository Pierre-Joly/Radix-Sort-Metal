import Foundation
import Metal

extension MetalRadixSorter {
    // MARK: - Public API: Array/List Convenience

    public func sort(_ values: [UInt32]) throws -> [UInt32] {
        guard !values.isEmpty else {
            return []
        }

        let buffer = try makeSharedUIntBuffer(values, label: "input buffer")
        try sort(buffer: buffer, count: values.count)
        return readArray(from: buffer, count: values.count, as: UInt32.self)
    }

    public func sortInPlace(_ values: inout [UInt32]) throws {
        values = try sort(values)
    }

    public func sortWithIndices(_ values: [UInt32]) throws -> IndexedSortResult {
        guard !values.isEmpty else {
            return IndexedSortResult(values: [], indices: [])
        }

        let valuesBuffer = try makeSharedUIntBuffer(values, label: "indexed input buffer")
        let indexBuffer = try makeSharedUIntBuffer(count: values.count, label: "indexed output buffer")

        try sort(buffer: valuesBuffer, indexBuffer: indexBuffer, count: values.count, initializeIndices: true)

        return IndexedSortResult(
            values: readArray(from: valuesBuffer, count: values.count, as: UInt32.self),
            indices: readArray(from: indexBuffer, count: values.count, as: UInt32.self)
        )
    }

    public func sortWithIndicesInPlace(_ values: inout [UInt32]) throws -> [UInt32] {
        let result = try sortWithIndices(values)
        values = result.values
        return result.indices
    }

    public func reorder<T>(values: [T], indices: [UInt32]) throws -> [T] {
        guard values.count == indices.count else {
            throw SortError.mismatchedElementCount(expected: values.count, got: indices.count)
        }
        guard !values.isEmpty else {
            return []
        }

        let count = values.count
        let elementStride = MemoryLayout<T>.stride
        let valueByteCount = try checkedByteCount(count: count, stride: elementStride)

        let valueBuffer = try makeSharedBuffer(from: values, byteCount: valueByteCount, label: "array reorder values buffer")
        let indexBuffer = try makeSharedUIntBuffer(indices, label: "array reorder index buffer")

        try reorder(buffer: valueBuffer, indexBuffer: indexBuffer, elementStride: elementStride, count: count)
        return readArray(from: valueBuffer, count: count, as: T.self)
    }

    public func reorderInPlace<T>(_ values: inout [T], indices: [UInt32]) throws {
        values = try reorder(values: values, indices: indices)
    }
}
