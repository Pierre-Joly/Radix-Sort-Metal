#ifndef RADIX_SORT_METAL_KERNEL_COMMON
#define RADIX_SORT_METAL_KERNEL_COMMON

#include <metal_stdlib>
using namespace metal;

#define kRadix 256u
#define kElementsPerBlock 2048u
#define kThreadgroupWidth 256u

#endif

kernel void initialize_indices(
    device uint *indices [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        indices[gid] = gid;
    }
}

kernel void clear_histogram_256(
    device uint *histogram [[buffer(0)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    histogram[lane] = 0u;
}

kernel void copy_uint_buffer(
    device const uint *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        destination[gid] = source[gid];
    }
}

kernel void copy_byte_buffer(
    device const uchar *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    constant uint &byteCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < byteCount) {
        destination[gid] = source[gid];
    }
}

kernel void reorder_by_index_bytes(
    device const uchar *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    device const uint *indices [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &elementStride [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    const uint destinationBase = gid * elementStride;
    const uint sourceIndex = indices[gid];

    if (sourceIndex >= count) {
        for (uint byteOffset = 0; byteOffset < elementStride; ++byteOffset) {
            destination[destinationBase + byteOffset] = 0u;
        }
        return;
    }

    const uint sourceBase = sourceIndex * elementStride;
    for (uint byteOffset = 0; byteOffset < elementStride; ++byteOffset) {
        destination[destinationBase + byteOffset] = source[sourceBase + byteOffset];
    }
}
