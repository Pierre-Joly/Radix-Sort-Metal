# RadixSortMetal

GPU radix sort for `UInt32` values on Apple GPUs (Metal), packaged as a Swift library.

## Features

- Stable 32-bit radix sort on GPU (4 passes, 8 bits per pass).
- Public API for sorting Swift arrays or existing `MTLBuffer` instances.
- Optional permutation-index output to reorder associated buffers/payloads.
- XCTest suite validating correctness on fixed and random inputs.

## Add To Another Project

Use Swift Package Manager and add this repository as a dependency:

```swift
.package(url: "https://github.com/<your-org>/RadixSortMetal.git", branch: "main")
```

Then depend on product `RadixSortMetal`.

## Usage

Sort a Swift array with the convenience API (input/output are CPU-side arrays, sorting runs on GPU):

```swift
import RadixSortMetal

let sorter = try MetalRadixSorter()
let sorted = try sorter.sort([9, 1, 4, 7, 3])
```

Get sorted keys plus permutation indices (`indices[newPosition] = originalPosition`):

```swift
import RadixSortMetal

let keys: [UInt32] = [50, 10, 50, 1]
let result = try MetalRadixSorter().sortWithIndices(keys)
// result.values  -> [1, 10, 50, 50]
// result.indices -> [3, 1, 0, 2]
```

Sort an existing Metal buffer in place:

```swift
import Metal
import RadixSortMetal

let device = MTLCreateSystemDefaultDevice()!
let sorter = try MetalRadixSorter(device: device)
let values: [UInt32] = [12, 5, 42, 1]
let buffer = device.makeBuffer(bytes: values,
                               length: values.count * MemoryLayout<UInt32>.stride,
                               options: .storageModeShared)!

try sorter.sort(buffer: buffer, count: values.count)
```

Sort an existing key buffer and get permutation in a second buffer:

```swift
let indexBuffer = device.makeBuffer(length: values.count * MemoryLayout<UInt32>.stride,
                                    options: .storageModeShared)!
try sorter.sort(buffer: buffer, indexBuffer: indexBuffer, count: values.count)
```

Append radix-sort work to an existing `MTLCommandBuffer`:

```swift
let commandBuffer = commandQueue.makeCommandBuffer()!
try sorter.encodeSort(buffer: buffer,
                      indexBuffer: indexBuffer,
                      count: values.count,
                      into: commandBuffer)
// encode other work, then commit once
commandBuffer.commit()
```

Or append directly to an already-open `MTLComputeCommandEncoder`:

```swift
let encoder = commandBuffer.makeComputeCommandEncoder()!
// existing compute dispatches...
try sorter.encodeSort(buffer: buffer, count: values.count, using: encoder)
// more compute dispatches...
encoder.endEncoding()
```

## Run Tests

```bash
swift test
```
