import Foundation

public struct IndexedSortResult {
    public let values: [UInt32]
    public let indices: [UInt32]

    public init(values: [UInt32], indices: [UInt32]) {
        self.values = values
        self.indices = indices
    }
}
