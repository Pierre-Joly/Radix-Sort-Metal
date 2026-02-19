import Metal
import XCTest
@testable import RadixSortMetal

extension MetalRadixSorterTests {
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

    func testSortWithIndicesReturnsPermutation() throws {
        let sorter = try makeSorter()
        let keys: [UInt32] = [50, 10, 50, 1, 10]
        let payload: [UInt32] = [500, 100, 501, 10, 101]

        let result = try sorter.sortWithIndices(keys)
        XCTAssertEqual(result.values, [1, 10, 10, 50, 50])

        let reorderedPayload = result.indices.map { payload[Int($0)] }
        XCTAssertEqual(reorderedPayload, [10, 100, 101, 500, 501])
    }

    func testSortWithIndicesInPlaceReturnsPermutation() throws {
        let sorter = try makeSorter()
        var keys: [UInt32] = [8, 4, 2, 9, 2]
        let original = keys

        let indices = try sorter.sortWithIndicesInPlace(&keys)
        XCTAssertEqual(keys, [2, 2, 4, 8, 9])
        XCTAssertEqual(indices.map { original[Int($0)] }, keys)
    }

    func testArrayReorderWithIndicesAndMultiplePayloads() throws {
        let sorter = try makeSorter()
        let keys: [UInt32] = [9, 3, 12, 3, 1]
        let payloadA: [UInt32] = [90, 30, 120, 31, 10]
        let payloadB: [Payload] = [
            Payload(id: 0, value: 900),
            Payload(id: 1, value: 300),
            Payload(id: 2, value: 1200),
            Payload(id: 3, value: 301),
            Payload(id: 4, value: 100)
        ]

        let result = try sorter.sortWithIndices(keys)
        XCTAssertEqual(result.values, [1, 3, 3, 9, 12])

        let reorderedA = try sorter.reorder(values: payloadA, indices: result.indices)
        XCTAssertEqual(reorderedA, [10, 30, 31, 90, 120])

        var reorderedB = payloadB
        try sorter.reorderInPlace(&reorderedB, indices: result.indices)
        XCTAssertEqual(reorderedB.map(\.id), [4, 1, 3, 0, 2])
    }
}
