import Foundation
import Metal
import RadixSortMetal

struct RadixSortBenchmarkMain {
    static func main() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            print("Metal device not available; benchmark skipped.")
            return
        }

        let outputDirectory = try parseOutputDirectory()
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let sorter = try MetalRadixSorter()
        var generator = LCG(seed: 0xDEAD_BEEF_CAFE_BABE)

        let sizes: [Int] = [
            512,
            1_024,
            2_048,
            4_096,
            8_192,
            16_384,
            32_768,
            65_536,
            131_072,
            262_144,
            524_288,
            1_048_576,
            2_097_152,
            4_194_304,
            8_388_608,
            16_777_216
        ]

        var rows: [BenchmarkRow] = []
        for size in sizes {
            let repetitions = repetitions(for: size)
            let warmups = 3
            var cpuTimes: [Double] = []
            var gpuTimes: [Double] = []

            print("Benchmarking size \(size) with \(repetitions) repetitions...")

            for repetition in 0..<(repetitions + warmups) {
                let values = randomValues(count: size, generator: &generator)
                if repetition < warmups {
                    _ = values.sorted()
                    _ = try sorter.sort(values)
                    continue
                }

                let (cpuSorted, cpuMs) = measure {
                    values.sorted()
                }
                let (gpuSorted, gpuMs) = try measure {
                    try sorter.sort(values)
                }

                guard cpuSorted == gpuSorted else {
                    throw BenchmarkError.inconsistentSort(size: size, repetition: repetition - warmups)
                }

                cpuTimes.append(cpuMs)
                gpuTimes.append(gpuMs)
            }

            let row = BenchmarkRow(
                size: size,
                repetitions: repetitions,
                cpuMedianMs: median(cpuTimes),
                cpuMeanMs: mean(cpuTimes),
                gpuMedianMs: median(gpuTimes),
                gpuMeanMs: mean(gpuTimes)
            )
            rows.append(row)

            let speedup = String(format: "%.2fx", row.speedup)
            print(
                "  CPU median: \(format(row.cpuMedianMs)) ms | GPU median: \(format(row.gpuMedianMs)) ms | speedup (CPU/GPU): \(speedup)"
            )
        }

        let csvURL = outputDirectory.appendingPathComponent("benchmark_results.csv")
        try writeCSV(rows: rows, to: csvURL)

        let plotURL = outputDirectory.appendingPathComponent("benchmark_plot.svg")
        let deviceName = MTLCreateSystemDefaultDevice()?.name ?? "Unknown Metal Device"
        try writeSVG(rows: rows, deviceName: deviceName, to: plotURL)

        let summaryURL = outputDirectory.appendingPathComponent("benchmark_summary.md")
        try writeSummary(rows: rows, deviceName: deviceName, to: summaryURL)

        print("Wrote benchmark CSV: \(csvURL.path)")
        print("Wrote benchmark plot: \(plotURL.path)")
        print("Wrote benchmark summary: \(summaryURL.path)")
    }

    static func parseOutputDirectory() throws -> URL {
        let args = Array(CommandLine.arguments.dropFirst())
        if args.isEmpty {
            return URL(fileURLWithPath: "Benchmarks/results", isDirectory: true)
        }

        if args.count == 2, args[0] == "--output" {
            return URL(fileURLWithPath: args[1], isDirectory: true)
        }

        if args.count == 1 {
            return URL(fileURLWithPath: args[0], isDirectory: true)
        }

        throw BenchmarkError.missingOutputArgument
    }
}

do {
    try RadixSortBenchmarkMain.main()
} catch {
    fatalError("Benchmark failed: \(error)")
}
