# `reorder`

## Summary

Reorder payload data from permutation indices (`indices[newPosition] = originalPosition`).

## Call Forms

- `reorder<T>(values: [T], indices: [UInt32]) throws -> [T]`
- `reorder(buffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int) throws`
- `reorder(buffer: MTLBuffer, temporaryBuffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int) throws`
- `reorder(sourceBuffer: MTLBuffer, destinationBuffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int) throws`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `values` | `[T]` | - | CPU payload values (array form). |
| `indices` | `[UInt32]` | - | CPU permutation indices (array form). |
| `buffer` | `MTLBuffer` | - | In-place GPU payload buffer. |
| `temporaryBuffer` | `MTLBuffer` | omitted | Reusable scratch buffer for in-place GPU reorder. |
| `sourceBuffer` | `MTLBuffer` | - | Source GPU payload buffer (out-of-place form). |
| `destinationBuffer` | `MTLBuffer` | - | Destination GPU payload buffer (out-of-place form). |
| `indexBuffer` | `MTLBuffer` | - | GPU permutation buffer (`UInt32`). |
| `elementStride` | `Int` | - | Payload element stride in bytes. |
| `count` | `Int` | - | Number of payload elements. |

## Returns

- Array form returns reordered payload (`[T]`).
- Buffer forms reorder GPU buffers and return `Void`.

## Default Behavior

- In-place buffer reorder without `temporaryBuffer` allocates transient scratch per call: `buffer -> scratch -> buffer`.
- In-place buffer reorder with `temporaryBuffer` reuses caller scratch: `buffer -> temporaryBuffer -> buffer`.
- Out-of-place buffer reorder is a single gather pass: `sourceBuffer -> destinationBuffer`.

## Examples

```swift
let reordered = try sorter.reorder(values: payload, indices: indices)
```

```swift
try sorter.reorder(
    buffer: payloadBuffer,
    indexBuffer: indexBuffer,
    elementStride: MemoryLayout<SIMD3<Float>>.stride,
    count: particleCount
)
```

```swift
try sorter.reorder(
    buffer: payloadBuffer,
    temporaryBuffer: payloadScratchBuffer,
    indexBuffer: indexBuffer,
    elementStride: MemoryLayout<SIMD3<Float>>.stride,
    count: particleCount
)
```
