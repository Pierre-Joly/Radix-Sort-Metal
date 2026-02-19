# RadixSortMetal Documentation

This is the global documentation entry point for `RadixSortMetal`.

## What This Library Provides

- Stable `UInt32` radix sort on Metal.
- Optional permutation index output.
- Reorder operations to apply the permutation to one or many payload buffers/arrays.
- Three integration modes:
  - Direct call (library creates/commits command buffer)
  - Append to existing `MTLCommandBuffer`
  - Append to existing `MTLComputeCommandEncoder`

## Reorder Model

- In-place reorder APIs use gather + copy-back.
- If no temporary buffer is provided, a transient scratch buffer is allocated for the call.
- If a temporary buffer is provided, that buffer is used as reusable scratch.
- Out-of-place reorder (`sourceBuffer` -> `destinationBuffer`) is a single gather pass with no copy-back.

## API Reference

### Constructors

- [`MetalRadixSorter.init(device:)`](Documentation/API/metal-radix-sorter-init.md)
- [`IndexedSortResult.init(values:indices:)`](Documentation/API/indexed-sort-result-init.md)

### Sort Functions

- [`sort`](Documentation/API/sort.md)
- [`sortInPlace`](Documentation/API/sort-in-place.md)
- [`sortWithIndices`](Documentation/API/sort-with-indices.md)
- [`sortWithIndicesInPlace`](Documentation/API/sort-with-indices-in-place.md)

### Reorder Functions

- [`reorder`](Documentation/API/reorder.md)
- [`reorderInPlace`](Documentation/API/reorder-in-place.md)

### Encoding Functions

- [`encodeSort`](Documentation/API/encode-sort.md)
- [`encodeReorder`](Documentation/API/encode-reorder.md)
