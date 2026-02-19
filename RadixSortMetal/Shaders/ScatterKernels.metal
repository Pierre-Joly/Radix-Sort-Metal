#ifndef RADIX_SORT_METAL_KERNEL_COMMON
#define RADIX_SORT_METAL_KERNEL_COMMON

#include <metal_stdlib>
using namespace metal;

#define kRadix 256u
#define kElementsPerBlock 2048u
#define kThreadgroupWidth 256u

#endif

kernel void scatter_values_stable(
    device const uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    device const uint *blockOffsets [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    uint lane [[thread_position_in_threadgroup]],
    uint blockIndex [[threadgroup_position_in_grid]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]]
) {
    if (threadsPerThreadgroup != kThreadgroupWidth) {
        return;
    }

    const uint start = blockIndex * kElementsPerBlock;
    if (start >= count) {
        return;
    }
    const uint end = min(start + kElementsPerBlock, count);

    threadgroup uint runningOffsets[kRadix];
    threadgroup uint tileStarts[kRadix];
    threadgroup uint tileDigits[kThreadgroupWidth];
    threadgroup uint tileValues[kThreadgroupWidth];
    threadgroup uchar tileValid[kThreadgroupWidth];

    for (uint bin = lane; bin < kRadix; bin += kThreadgroupWidth) {
        runningOffsets[bin] = blockOffsets[blockIndex * kRadix + bin];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tileStart = start; tileStart < end; tileStart += kThreadgroupWidth) {
        const uint index = tileStart + lane;
        const bool valid = index < end;

        if (valid) {
            const uint value = input[index];
            tileValues[lane] = value;
            tileDigits[lane] = (value >> shift) & 0xFFu;
            tileValid[lane] = 1u;
        } else {
            tileValues[lane] = 0u;
            tileDigits[lane] = 0u;
            tileValid[lane] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            for (uint bin = 0; bin < kRadix; ++bin) {
                tileStarts[bin] = runningOffsets[bin];
            }

            for (uint localIndex = 0; localIndex < kThreadgroupWidth; ++localIndex) {
                if (tileValid[localIndex] != 0u) {
                    const uint digit = tileDigits[localIndex];
                    runningOffsets[digit] += 1u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tileValid[lane] != 0u) {
            const uint digit = tileDigits[lane];
            uint rank = 0u;
            for (uint previous = 0; previous < lane; ++previous) {
                if (tileValid[previous] != 0u && tileDigits[previous] == digit) {
                    rank += 1u;
                }
            }
            const uint destination = tileStarts[digit] + rank;
            output[destination] = tileValues[lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void scatter_key_index_stable(
    device const uint *inputValues [[buffer(0)]],
    device uint *outputValues [[buffer(1)]],
    device const uint *inputIndices [[buffer(2)]],
    device uint *outputIndices [[buffer(3)]],
    device const uint *blockOffsets [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    constant uint &shift [[buffer(6)]],
    uint lane [[thread_position_in_threadgroup]],
    uint blockIndex [[threadgroup_position_in_grid]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]]
) {
    if (threadsPerThreadgroup != kThreadgroupWidth) {
        return;
    }

    const uint start = blockIndex * kElementsPerBlock;
    if (start >= count) {
        return;
    }
    const uint end = min(start + kElementsPerBlock, count);

    threadgroup uint runningOffsets[kRadix];
    threadgroup uint tileStarts[kRadix];
    threadgroup uint tileDigits[kThreadgroupWidth];
    threadgroup uint tileValues[kThreadgroupWidth];
    threadgroup uint tileIndices[kThreadgroupWidth];
    threadgroup uchar tileValid[kThreadgroupWidth];

    for (uint bin = lane; bin < kRadix; bin += kThreadgroupWidth) {
        runningOffsets[bin] = blockOffsets[blockIndex * kRadix + bin];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tileStart = start; tileStart < end; tileStart += kThreadgroupWidth) {
        const uint index = tileStart + lane;
        const bool valid = index < end;

        if (valid) {
            const uint value = inputValues[index];
            tileValues[lane] = value;
            tileDigits[lane] = (value >> shift) & 0xFFu;
            tileIndices[lane] = inputIndices[index];
            tileValid[lane] = 1u;
        } else {
            tileValues[lane] = 0u;
            tileDigits[lane] = 0u;
            tileIndices[lane] = 0u;
            tileValid[lane] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            for (uint bin = 0; bin < kRadix; ++bin) {
                tileStarts[bin] = runningOffsets[bin];
            }

            for (uint localIndex = 0; localIndex < kThreadgroupWidth; ++localIndex) {
                if (tileValid[localIndex] != 0u) {
                    const uint digit = tileDigits[localIndex];
                    runningOffsets[digit] += 1u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tileValid[lane] != 0u) {
            const uint digit = tileDigits[lane];
            uint rank = 0u;
            for (uint previous = 0; previous < lane; ++previous) {
                if (tileValid[previous] != 0u && tileDigits[previous] == digit) {
                    rank += 1u;
                }
            }
            const uint destination = tileStarts[digit] + rank;
            outputValues[destination] = tileValues[lane];
            outputIndices[destination] = tileIndices[lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
