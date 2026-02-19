# `MetalRadixSorter.init(device:)`

## Summary

Create a sorter instance and initialize Metal pipelines.

## Call Forms

- `init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws`

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `device` | `MTLDevice?` | `MTLCreateSystemDefaultDevice()` | Metal device used to create command queue, library, and pipelines. |

## Returns

- A configured `MetalRadixSorter` instance.

## Default Behavior

- If `device` is omitted, the system default Metal device is used.

## Examples

```swift
let sorter = try MetalRadixSorter()
```

```swift
let sorter2 = try MetalRadixSorter(device: MTLCreateSystemDefaultDevice())
```
