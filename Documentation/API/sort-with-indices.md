# `sortWithIndices`

## Summary

Sort keys and return both sorted keys and permutation indices.

## Call Forms

- `sortWithIndices(_ values: [UInt32]) throws -> IndexedSortResult`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `values` | `[UInt32]` | - | Input keys. |

## Returns

- `IndexedSortResult` with:
  - `values`: sorted keys.
  - `indices`: permutation where `indices[newPosition] = originalPosition`.

## Default Behavior

- Empty input returns `IndexedSortResult(values: [], indices: [])`.

## Examples

```swift
let result = try sorter.sortWithIndices([9, 3, 12, 3, 1])
```
