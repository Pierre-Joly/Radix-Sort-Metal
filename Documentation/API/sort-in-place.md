# `sortInPlace`

## Summary

Sort an array of `UInt32` keys in place.

## Call Forms

- `sortInPlace(_ values: inout [UInt32]) throws`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `values` | `inout [UInt32]` | - | Input keys, sorted in place. |

## Returns

- Returns `Void`.

## Default Behavior

- Empty arrays are a no-op.

## Examples

```swift
var keys: [UInt32] = [9, 3, 12, 3, 1]
try sorter.sortInPlace(&keys)
```
