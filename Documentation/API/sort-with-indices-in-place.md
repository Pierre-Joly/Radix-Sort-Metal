# `sortWithIndicesInPlace`

## Summary

Sort keys in place and return permutation indices.

## Call Forms

- `sortWithIndicesInPlace(_ values: inout [UInt32]) throws -> [UInt32]`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `values` | `inout [UInt32]` | - | Input keys, sorted in place. |

## Returns

- Permutation indices where `indices[newPosition] = originalPosition`.

## Default Behavior

- Empty input returns `[]` and leaves `values` empty.

## Examples

```swift
var keys: [UInt32] = [9, 3, 12, 3, 1]
let indices = try sorter.sortWithIndicesInPlace(&keys)
```
