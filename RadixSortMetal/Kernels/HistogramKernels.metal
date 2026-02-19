#ifndef RADIX_SORT_METAL_KERNEL_COMMON
#define RADIX_SORT_METAL_KERNEL_COMMON

#include <metal_stdlib>
using namespace metal;

#define kRadix 256u
#define kElementsPerBlock 2048u
#define kThreadgroupWidth 256u

#endif

kernel void count_block_histograms(
    device const uint *input [[buffer(0)]],
    device uint *blockHistograms [[buffer(1)]],
    device atomic_uint *totalHistogram [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    uint lane [[thread_position_in_threadgroup]],
    uint blockIndex [[threadgroup_position_in_grid]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint localHistogram[kRadix];

    for (uint bin = lane; bin < kRadix; bin += threadsPerThreadgroup) {
        atomic_store_explicit(&localHistogram[bin], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start = blockIndex * kElementsPerBlock;
    const uint end = min(start + kElementsPerBlock, count);

    for (uint index = start + lane; index < end; index += threadsPerThreadgroup) {
        const uint value = input[index];
        const uint digit = (value >> shift) & 0xFFu;
        atomic_fetch_add_explicit(&localHistogram[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint bin = lane; bin < kRadix; bin += threadsPerThreadgroup) {
        const uint localValue = atomic_load_explicit(&localHistogram[bin], memory_order_relaxed);
        const uint flatIndex = blockIndex * kRadix + bin;
        blockHistograms[flatIndex] = localValue;

        if (localValue != 0u) {
            atomic_fetch_add_explicit(&totalHistogram[bin], localValue, memory_order_relaxed);
        }
    }
}

kernel void scan_total_histogram(
    device const uint *totalHistogram [[buffer(0)]],
    device uint *binOffsets [[buffer(1)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup uint scratch[kRadix];
    scratch[lane] = totalHistogram[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < kRadix; stride <<= 1) {
        const uint index = ((lane + 1u) * stride * 2u) - 1u;
        if (index < kRadix) {
            scratch[index] += scratch[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        scratch[kRadix - 1u] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int stride = int(kRadix / 2u); stride > 0; stride >>= 1) {
        const uint strideValue = uint(stride);
        const uint index = ((lane + 1u) * strideValue * 2u) - 1u;

        if (index < kRadix) {
            const uint left = scratch[index - strideValue];
            const uint right = scratch[index];
            scratch[index - strideValue] = right;
            scratch[index] = right + left;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    binOffsets[lane] = scratch[lane];
}

kernel void build_block_offsets(
    device const uint *blockHistograms [[buffer(0)]],
    device const uint *binOffsets [[buffer(1)]],
    device uint *blockOffsets [[buffer(2)]],
    constant uint &blockCount [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint running = binOffsets[lane];

    for (uint block = 0; block < blockCount; ++block) {
        const uint flatIndex = block * kRadix + lane;
        blockOffsets[flatIndex] = running;
        running += blockHistograms[flatIndex];
    }
}
