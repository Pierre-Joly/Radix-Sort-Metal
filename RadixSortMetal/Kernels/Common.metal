#ifndef RADIX_SORT_METAL_KERNEL_COMMON
#define RADIX_SORT_METAL_KERNEL_COMMON

#include <metal_stdlib>
using namespace metal;

#define kRadix 256u
#define kElementsPerBlock 2048u
#define kThreadgroupWidth 256u

#endif
