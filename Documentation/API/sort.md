# `sort`

## Summary

Sort `UInt32` keys using GPU radix sort.

## Call Forms

- `sort(_ values: [UInt32]) throws -> [UInt32]`
- `sort(buffer: MTLBuffer, count: Int) throws`
- `sort(buffer: MTLBuffer, indexBuffer: MTLBuffer, count: Int, initializeIndices: Bool = true) throws`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `values` | `[UInt32]` | - | CPU input keys (array form). |
| `buffer` | `MTLBuffer` | - | GPU key buffer containing `UInt32` values. |
| `indexBuffer` | `MTLBuffer` | required in indexed form | GPU output permutation buffer (`UInt32`). |
| `count` | `Int` | - | Number of key elements in `buffer`. |
| `initializeIndices` | `Bool` | `true` | Whether to write identity indices before sorting. |

## Returns

- Array form returns sorted values (`[UInt32]`).
- Buffer forms sort in place and return `Void`.

## Default Behavior

- Omitting `initializeIndices` uses `true`.
- With `initializeIndices == true`, `indexBuffer` is initialized to `0..<count`.
- With `initializeIndices == false`, existing `indexBuffer` values are used as input permutation state.

## Examples

```swift
let sorted = try sorter.sort([9, 3, 12, 3, 1])
```

```swift
try sorter.sort(buffer: keyBuffer, count: keyCount)
```

```swift
try sorter.sort(buffer: keyBuffer, indexBuffer: indexBuffer, count: keyCount)
```
