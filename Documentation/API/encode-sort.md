# `encodeSort`

## Summary

Encode sort work into an existing command context (`MTLCommandBuffer` or `MTLComputeCommandEncoder`).

## Call Forms

- `encodeSort(buffer: MTLBuffer, count: Int, into commandBuffer: MTLCommandBuffer) throws`
- `encodeSort(buffer: MTLBuffer, indexBuffer: MTLBuffer, count: Int, initializeIndices: Bool = true, into commandBuffer: MTLCommandBuffer) throws`
- `encodeSort(buffer: MTLBuffer, count: Int, using encoder: MTLComputeCommandEncoder) throws`
- `encodeSort(buffer: MTLBuffer, indexBuffer: MTLBuffer, count: Int, initializeIndices: Bool = true, using encoder: MTLComputeCommandEncoder) throws`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `buffer` | `MTLBuffer` | - | GPU key buffer (`UInt32`). |
| `indexBuffer` | `MTLBuffer` | required in indexed form | GPU permutation output buffer (`UInt32`). |
| `count` | `Int` | - | Number of key elements. |
| `initializeIndices` | `Bool` | `true` | Whether to write identity indices before sorting. |
| `commandBuffer` | `MTLCommandBuffer` | - | Destination command buffer for encoding. |
| `encoder` | `MTLComputeCommandEncoder` | - | Destination compute encoder for encoding. |

## Returns

- Returns `Void`.

## Default Behavior

- Omitting `initializeIndices` uses `true`.
- With `initializeIndices == true`, identity permutation is encoded before sort.

## Examples

```swift
try sorter.encodeSort(buffer: keyBuffer, count: keyCount, into: commandBuffer)
```

```swift
try sorter.encodeSort(buffer: keyBuffer, indexBuffer: indexBuffer, count: keyCount, using: encoder)
```
