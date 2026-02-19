import Foundation

extension RadixSortBenchmarkMain {
    static func writeCSV(rows: [BenchmarkRow], to url: URL) throws {
        var lines: [String] = []
        lines.append("size,repetitions,cpu_median_ms,cpu_mean_ms,gpu_median_ms,gpu_mean_ms,speedup_cpu_over_gpu")
        for row in rows {
            lines.append(
                "\(row.size),\(row.repetitions),\(format(row.cpuMedianMs)),\(format(row.cpuMeanMs)),\(format(row.gpuMedianMs)),\(format(row.gpuMeanMs)),\(format(row.speedup))"
            )
        }
        try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    static func writeSummary(rows: [BenchmarkRow], deviceName: String, to url: URL) throws {
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        var lines: [String] = []
        lines.append("# Benchmark Summary")
        lines.append("")
        lines.append("- Device: \(deviceName)")
        lines.append("- OS: \(os)")
        lines.append("- Date: \(ISO8601DateFormatter().string(from: Date()))")
        lines.append("")
        lines.append("| Size | Repetitions | CPU median (ms) | GPU median (ms) | CPU/GPU speedup |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in rows {
            lines.append(
                "| \(row.size) | \(row.repetitions) | \(format(row.cpuMedianMs)) | \(format(row.gpuMedianMs)) | \(format(row.speedup))x |"
            )
        }
        try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    static func writeSVG(rows: [BenchmarkRow], deviceName: String, to url: URL) throws {
        let width = 1100.0
        let height = 700.0
        let left = 100.0
        let right = 40.0
        let top = 90.0
        let bottom = 80.0
        let plotWidth = width - left - right
        let plotHeight = height - top - bottom

        let sizes = rows.map { Double($0.size) }
        let cpu = rows.map { $0.cpuMedianMs }
        let gpu = rows.map { $0.gpuMedianMs }
        let allTimes = cpu + gpu

        let xMin = Foundation.log2(sizes.min() ?? 1.0)
        let xMax = Foundation.log2(sizes.max() ?? 1.0)
        let yMinLog = floor(Foundation.log10((allTimes.min() ?? 0.001) * 0.8))
        let yMaxLog = ceil(Foundation.log10((allTimes.max() ?? 1.0) * 1.25))

        func mapX(_ size: Double) -> Double {
            let t = (Foundation.log2(size) - xMin) / (xMax - xMin)
            return left + (t * plotWidth)
        }

        func mapY(_ time: Double) -> Double {
            let t = (Foundation.log10(time) - yMinLog) / (yMaxLog - yMinLog)
            return top + ((1.0 - t) * plotHeight)
        }

        func linePath(values: [Double]) -> String {
            var segments: [String] = []
            for (index, value) in values.enumerated() {
                let x = mapX(sizes[index])
                let y = mapY(value)
                segments.append(index == 0 ? "M \(x) \(y)" : "L \(x) \(y)")
            }
            return segments.joined(separator: " ")
        }

        let cpuPath = linePath(values: cpu)
        let gpuPath = linePath(values: gpu)
        let slopeRows = rows.filter { $0.size >= 4_096 }
        let slopeInput = slopeRows.count >= 2 ? slopeRows : rows
        let slopeSizes = slopeInput.map { Double($0.size) }
        let slopeCPU = slopeInput.map { $0.cpuMedianMs }
        let slopeGPU = slopeInput.map { $0.gpuMedianMs }
        let cpuSlope = slope(of: slopeSizes, and: slopeCPU)
        let gpuSlope = slope(of: slopeSizes, and: slopeGPU)

        var grid: [String] = []
        for row in rows {
            let x = mapX(Double(row.size))
            grid.append("<line x1='\(x)' y1='\(top)' x2='\(x)' y2='\(top + plotHeight)' stroke='#ececec' stroke-width='1' />")
            grid.append("<text x='\(x)' y='\(top + plotHeight + 24)' text-anchor='middle' font-size='12' fill='#555'>\(row.size)</text>")
        }

        for exp in Int(yMinLog)...Int(yMaxLog) {
            let value = Foundation.pow(10.0, Double(exp))
            let y = mapY(value)
            grid.append("<line x1='\(left)' y1='\(y)' x2='\(left + plotWidth)' y2='\(y)' stroke='#ececec' stroke-width='1' />")
            grid.append("<text x='\(left - 12)' y='\(y + 4)' text-anchor='end' font-size='12' fill='#555'>\(format(value)) ms</text>")
        }

        let cpuPoints = zip(sizes, cpu).map { size, time -> String in
            let x = mapX(size)
            let y = mapY(time)
            return "<circle cx='\(x)' cy='\(y)' r='4' fill='#D95F02' />"
        }.joined(separator: "\n")

        let gpuPoints = zip(sizes, gpu).map { size, time -> String in
            let x = mapX(size)
            let y = mapY(time)
            return "<circle cx='\(x)' cy='\(y)' r='4' fill='#1B9E77' />"
        }.joined(separator: "\n")

        let legendX = left + 14.0
        let legendY = top + 14.0
        let legendWidth = 285.0
        let legendHeight = 86.0
        let legendSwatchStartX = legendX + 16.0
        let legendSwatchEndX = legendX + 66.0
        let legendTextX = legendX + 76.0

        let svg = """
        <svg xmlns='http://www.w3.org/2000/svg' width='\(Int(width))' height='\(Int(height))' viewBox='0 0 \(Int(width)) \(Int(height))'>
          <rect width='100%' height='100%' fill='white' />
          <text x='\(width / 2)' y='38' text-anchor='middle' font-size='26' font-family='-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif' fill='#222'>
            RadixSortMetal vs Swift sorted() (Median Time)
          </text>
          <text x='\(width / 2)' y='62' text-anchor='middle' font-size='14' font-family='-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif' fill='#666'>
            Log-Log plot across input sizes (complexity trend). Device: \(escape(deviceName))
          </text>

          \(grid.joined(separator: "\n"))

          <rect x='\(left)' y='\(top)' width='\(plotWidth)' height='\(plotHeight)' fill='none' stroke='#777' stroke-width='1.2' />
          <path d='\(cpuPath)' fill='none' stroke='#D95F02' stroke-width='3' />
          <path d='\(gpuPath)' fill='none' stroke='#1B9E77' stroke-width='3' />
          \(cpuPoints)
          \(gpuPoints)

          <text x='\(left + plotWidth / 2)' y='\(height - 24)' text-anchor='middle' font-size='14' fill='#333'>Input size (number of UInt32 values)</text>
          <text transform='translate(24,\(top + plotHeight / 2)) rotate(-90)' text-anchor='middle' font-size='14' fill='#333'>Median time (ms, log scale)</text>

          <rect x='\(legendX)' y='\(legendY)' width='\(legendWidth)' height='\(legendHeight)' fill='white' stroke='#ddd' />
          <line x1='\(legendSwatchStartX)' y1='\(legendY + 26.0)' x2='\(legendSwatchEndX)' y2='\(legendY + 26.0)' stroke='#D95F02' stroke-width='3' />
          <text x='\(legendTextX)' y='\(legendY + 30.0)' font-size='13' fill='#333'>Swift sorted() (tail slope \(format(cpuSlope)))</text>
          <line x1='\(legendSwatchStartX)' y1='\(legendY + 54.0)' x2='\(legendSwatchEndX)' y2='\(legendY + 54.0)' stroke='#1B9E77' stroke-width='3' />
          <text x='\(legendTextX)' y='\(legendY + 58.0)' font-size='13' fill='#333'>Metal radix sort (tail slope \(format(gpuSlope)))</text>
        </svg>
        """

        try svg.write(to: url, atomically: true, encoding: .utf8)
    }
}
