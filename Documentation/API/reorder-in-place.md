# `reorderInPlace`

## Summary

Reorder a payload array in place using permutation indices.

## Call Forms

- `reorderInPlace<T>(_ values: inout [T], indices: [UInt32]) throws`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `values` | `inout [T]` | - | Payload values reordered in place. |
| `indices` | `[UInt32]` | - | Permutation where `indices[newPosition] = originalPosition`. |

## Returns

- Returns `Void`.

## Default Behavior

- None.

## Examples

```swift
var payload = [90, 30, 120, 31, 10]
let indices: [UInt32] = [4, 1, 3, 0, 2]
try sorter.reorderInPlace(&payload, indices: indices)
```
