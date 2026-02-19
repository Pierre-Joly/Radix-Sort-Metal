#include <metal_stdlib>
using namespace metal;

constant uint kRadix = 256;
constant uint kElementsPerBlock = 2048;
constant uint kThreadgroupWidth = 256;

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
