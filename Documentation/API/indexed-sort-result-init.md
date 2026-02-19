# `IndexedSortResult.init(values:indices:)`

## Summary

Create a container that stores sorted keys and their permutation indices.

## Call Forms

- `init(values: [UInt32], indices: [UInt32])`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `values` | `[UInt32]` | - | Sorted key array. |
| `indices` | `[UInt32]` | - | Permutation where `indices[newPosition] = originalPosition`. |

## Returns

- An `IndexedSortResult` value.

## Default Behavior

- None.

## Examples

```swift
let result = IndexedSortResult(values: [1, 3, 3, 9], indices: [2, 1, 3, 0])
```
