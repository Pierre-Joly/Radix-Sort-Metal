# `encodeReorder`

## Summary

Encode reorder work into an existing command context (`MTLCommandBuffer` or `MTLComputeCommandEncoder`).

## Call Forms

- `encodeReorder(buffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int, into commandBuffer: MTLCommandBuffer) throws`
- `encodeReorder(buffer: MTLBuffer, temporaryBuffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int, into commandBuffer: MTLCommandBuffer) throws`
- `encodeReorder(sourceBuffer: MTLBuffer, destinationBuffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int, into commandBuffer: MTLCommandBuffer) throws`
- `encodeReorder(buffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int, using encoder: MTLComputeCommandEncoder) throws`
- `encodeReorder(buffer: MTLBuffer, temporaryBuffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int, using encoder: MTLComputeCommandEncoder) throws`
- `encodeReorder(sourceBuffer: MTLBuffer, destinationBuffer: MTLBuffer, indexBuffer: MTLBuffer, elementStride: Int, count: Int, using encoder: MTLComputeCommandEncoder) throws`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `buffer` | `MTLBuffer` | - | In-place GPU payload buffer. |
| `temporaryBuffer` | `MTLBuffer` | omitted | Reusable scratch buffer for in-place reorder. |
| `sourceBuffer` | `MTLBuffer` | - | Source GPU payload buffer (out-of-place form). |
| `destinationBuffer` | `MTLBuffer` | - | Destination GPU payload buffer (out-of-place form). |
| `indexBuffer` | `MTLBuffer` | - | GPU permutation buffer (`UInt32`). |
| `elementStride` | `Int` | - | Payload element size in bytes. |
| `count` | `Int` | - | Number of payload elements. |
| `commandBuffer` | `MTLCommandBuffer` | - | Destination command buffer for encoding. |
| `encoder` | `MTLComputeCommandEncoder` | - | Destination compute encoder for encoding. |

## Returns

- Returns `Void`.

## Default Behavior

- In-place variants without `temporaryBuffer` allocate transient scratch per call: `buffer -> scratch -> buffer`.
- In-place variants with `temporaryBuffer` reuse caller scratch: `buffer -> temporaryBuffer -> buffer`.
- Source/destination variants are single-pass gather: `sourceBuffer -> destinationBuffer`.

## Examples

```swift
try sorter.encodeReorder(
    buffer: payloadBuffer,
    indexBuffer: indexBuffer,
    elementStride: MemoryLayout<SIMD3<Float>>.stride,
    count: particleCount,
    into: commandBuffer
)
```

```swift
try sorter.encodeReorder(
    buffer: payloadBuffer,
    temporaryBuffer: payloadScratchBuffer,
    indexBuffer: indexBuffer,
    elementStride: MemoryLayout<SIMD3<Float>>.stride,
    count: particleCount,
    using: encoder
)
```
