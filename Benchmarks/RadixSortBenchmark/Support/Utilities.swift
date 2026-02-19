import Foundation

extension RadixSortBenchmarkMain {
    static func randomValues(count: Int, generator: inout LCG) -> [UInt32] {
        var values: [UInt32] = []
        values.reserveCapacity(count)
        for _ in 0..<count {
            values.append(generator.next())
        }
        return values
    }

    static func repetitions(for size: Int) -> Int {
        if size >= 16_777_216 {
            return 3
        }
        if size >= 8_388_608 {
            return 4
        }
        return max(8, min(35, 2_000_000 / size))
    }

    static func measure<T>(_ block: () throws -> T) rethrows -> (T, Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        let result = try block()
        let end = DispatchTime.now().uptimeNanoseconds
        let elapsedMs = Double(end - start) / 1_000_000.0
        return (result, elapsedMs)
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let count = sorted.count
        if count.isMultiple(of: 2) {
            return (sorted[(count / 2) - 1] + sorted[count / 2]) * 0.5
        }
        return sorted[count / 2]
    }

    static func mean(_ values: [Double]) -> Double {
        values.reduce(0.0, +) / Double(values.count)
    }

    static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    static func slope(of xs: [Double], and ys: [Double]) -> Double {
        let n = Double(xs.count)
        let lx = xs.map(Foundation.log)
        let ly = ys.map(Foundation.log)
        let sumX = lx.reduce(0.0, +)
        let sumY = ly.reduce(0.0, +)
        let sumXX = zip(lx, lx).reduce(0.0) { $0 + ($1.0 * $1.1) }
        let sumXY = zip(lx, ly).reduce(0.0) { $0 + ($1.0 * $1.1) }

        let denominator = (n * sumXX) - (sumX * sumX)
        if denominator == 0.0 {
            return 0.0
        }
        return ((n * sumXY) - (sumX * sumY)) / denominator
    }

    static func escape(_ string: String) -> String {
        string
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "'", with: "&apos;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}
