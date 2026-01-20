# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import cupy as cp
import numpy as np

bin_stat_code = """
// CUDA kernel that performs parallel binned statistic computation.
/* For each input sample:
 *  - Compute the histogram bin index from its coordinate
 *  - Clamp the index to the valid bin range
 *  - Atomically increment the bin count
 *  - Atomically accumulate the corresponding value
 */
    __global__ void binned_statistic_kernel_v1(
        const int n_samples,
        const int bins,
        const double x_min,
        const double bin_width,
        const double * __restrict__ x,
        const double * __restrict__ values,
        unsigned long long * __restrict__ bin_counts,
        double * __restrict__ bin_sums
    )
    /* Parameters:
     *  n_samples     : Number of input samples
     *  bins        : Number of histogram bins
     *  x_min         : Lower bound of the histogram range
     *  bin_width     : Width of each histogram bin
     *  x             : Input coordinates (size n_samples)
     *  values        : Values to accumulate per bin (size n_samples)
     *  bin_counts    : Output histogram counts (size bins)
     *  bin_sums      : Output accumulated values per bin (size bins)
     */
    {
        // Global thread index
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // Guard against out-of-bounds threads
        if (tid < n_samples) {
            // Compute bin index from coordinate
            int bin_ibin_width = (int)((x[tid] - x_min) / bin_width);
            // Clamp bin index to valid range [0, bins - 1]
            bin_ibin_width = max(0, min(bin_ibin_width, bins - 1));
            // Atomically update histogram count
            atomicAdd(&bin_counts[bin_ibin_width], 1ULL);
            // Atomically accumulate value into the bin
            atomicAdd(&bin_sums[bin_ibin_width], values[tid]);
        }
    }





/* First Idea: Privatization
 *
 * Highly contended data structustatistic are replicated into private copies,
 * typically one per thread block. Each block updates its own copy,
 * reducing contention and lowering access latency.
 *
 * The private copies must be merged after computation, so performance
 * depends on balancing reduced contention with merge overhead.
 * Contention occurs only within a block and during the final merge.
 *
 * Each block:
 *  1. Accumulates a private histogram (counts and sums) in global memory
 *     indexed by [blockIdx.x * bins + bin].
 *  2. After synchronization, blocks with blockIdx.x > 0 reduce their
 *     private histograms into the global histogram at index [bin].
 */

    __global__ void binned_statistic_kernel_v2(
        const int n_samples,
        const int bins,
        const double x_min,
        const double bin_width,
        const double * __restrict__ x,
        const double * __restrict__ values,
        unsigned long long * __restrict__ block_counts,
        double * __restrict__ block_sums
    )
    /* Parameters:
     *  n_samples     : Number of input samples
     *  bins        : Number of histogram bins
     *  x_min         : Lower bound of the histogram range
     *  bin_width     : Width of each histogram bin
     *  x             : Input coordinates (size n_samples)
     *  values        : Values to accumulate per bin (size n_samples)
     *  block_counts  : Per-block histogram counts
     *                  (size gridDim.x * bins)
     *  block_sums    : Per-block histogram sums
     *                  (size gridDim.x * bins)
     *
     * Output:
     *  block 0 accumulates the final global histogram in
     *  block_counts[0..bins-1] and block_sums[0..bins-1]
     */
    {
        // Global thread index
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        /*------------------------------------------------------------
         * Phase 1: Per-block histogram accumulation
         *------------------------------------------------------------
         */
        if (tid < n_samples) {
            // Compute bin index for this sample
            int bin_ibin_width = (int)((x[tid] - x_min) / bin_width);
            // Clamp bin index to valid range [0, bins - 1]
            bin_ibin_width = max(0, min(bin_ibin_width, bins - 1));
            // Offset to this block's private histogram
            unsigned int block_offset = blockIdx.x * bins + bin_ibin_width;
            // Atomically update per-block histogram
            atomicAdd(&block_counts[block_offset], 1ULL);
            atomicAdd(&block_sums[block_offset], values[tid]);
        }
        // Ensure all threads in the block have finished updating
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 2: Reduction of per-block histograms
         *------------------------------------------------------------
         */
        if (blockIdx.x > 0) {
            // Threads cooperatively reduce this block's bins
            for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
                unsigned int block_offset = blockIdx.x * bins + bin;
                unsigned long long local_count = block_counts[block_offset];
                double local_sum = block_sums[block_offset];
                // Skip empty bins to reduce atomics
                if (local_count > 0) {
                    atomicAdd(&block_counts[bin], local_count);
                    atomicAdd(&block_sums[bin], local_sum);
                }
            }
        }
    }





/*
 * If the histogram has few bins, each block's private copy can be stored in shared memory.
 * Shared memory is block-private, so it cannot be shared across blocks.
 * Placing the histogram in shared memory greatly reduces access latency
 * and improves atomic operation throughput.
 * Below is a kernel that uses shared-memory private histograms.
 */

    __global__ void binned_statistic_kernel_v3(
        const int n_samples,
        const int bins,
        const double x_min,
        const double bin_width,
        const double * __restrict__ x,
        const double * __restrict__ values,
        unsigned long long * __restrict__ global_counts,
        double * __restrict__ global_sums
    )
    /* Parameters:
     *  n_samples     : Number of input samples
     *  bins        : Number of histogram bins
     *  x_min         : Lower bound of the histogram range
     *  bin_width     : Width of each histogram bin
     *  x             : Input coordinates (size n_samples)
     *  values        : Values to accumulate per bin (size n_samples)
     *  global_counts : Global histogram counts (size bins)
     *  global_sums   : Global accumulated values per bin (size bins)
     */
    {
        // Dynamically allocated shared memory
        extern __shared__ unsigned char smem[];
        // Shared-memory views
        unsigned long long* bin_counts_s = (unsigned long long*)smem;
        double* bin_sums_s = (double*)(smem + bins * sizeof(unsigned long long));
        /*------------------------------------------------------------
         * Phase 1: Initialize shared-memory histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            bin_counts_s[bin] = 0ULL;
            bin_sums_s[bin]  = 0.0;
        }
        // Ensure all bins are initialized
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 2: Accumulate samples into shared memory
         *------------------------------------------------------------
         */
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n_samples) {
            // Compute bin index
            int bin_ibin_width = (int)((x[tid] - x_min) / bin_width);
            bin_ibin_width = max(0, min(bin_ibin_width, bins - 1));
            // Atomic updates in shared memory
            atomicAdd(&bin_sums_s[bin_ibin_width], values[tid]);
            atomicAdd(&bin_counts_s[bin_ibin_width], 1ULL);
        }
        // Ensure all threads finished updating shared histograms
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 3: Merge block histograms into global histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            unsigned long long local_count = bin_counts_s[bin];
            double local_sum = bin_sums_s[bin];
            // Skip empty bins to reduce global atomics
            if (local_count > 0) {
                atomicAdd(&global_sums[bin],   local_sum);
                atomicAdd(&global_counts[bin], local_count);
            }
        }
    }





#define CFACTOR 2

/* Second Idea: Coarsening
 *
 * Privatization incurs overhead when private copies are merged back,
 * once per thread block. If too many blocks are launched, this overhead
 * may be wasted due to block serialization.
 *
 * Thread coarsening reduces this cost by using fewer blocks and letting
 * each thread process multiple input elements.
 *
 * With contiguous partitioning, each thread handles CFACTOR consecutive
 * elements, reducing the number of private copies while remaining simple
 * and intuitive.
 *
 * Each block:
 *  1. Initializes shared-memory histograms (counts and sums)
 *  2. Each thread processes CFACTOR consecutive samples
 *  3. Accumulates statisticults into shared memory
 *  4. Atomically merges the block histogram into the global histogram
 */
    __global__ void binned_statistic_kernel_v4(
        const int n_samples,
        const int bins,
        const double x_min,
        const double bin_width,
        const double * __restrict__ x,
        const double * __restrict__ values,
        unsigned long long * __restrict__ global_counts,
        double * __restrict__ global_sums
    )
    /* Parameters:
     *  n_samples     : Number of input samples
     *  bins        : Number of histogram bins
     *  x_min         : Lower bound of the histogram range
     *  bin_width     : Width of each histogram bin
     *  x             : Input coordinates (size n_samples)
     *  values        : Values to accumulate per bin (size n_samples)
     *  global_counts : Global histogram counts (size bins)
     *  global_sums   : Global accumulated values per bin (size bins)
     */
    {
        // Dynamically allocated shared memory (typed as double for alignment)
        extern __shared__ double shmem[];
        // Shared-memory views
        unsigned long long *bin_counts_s = reinterpret_cast<unsigned long long *>(shmem);
        double *bin_sums_s = reinterpret_cast<double *>(&bin_counts_s[bins]);
        /*------------------------------------------------------------
         * Phase 1: Initialize shared-memory histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            bin_counts_s[bin] = 0ULL;
            bin_sums_s[bin]   = 0.0;
        }
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 2: Thread-coarsened accumulation
         *------------------------------------------------------------
         */
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // Each thread processes CFACTOR consecutive samples
        for (unsigned int i = tid * CFACTOR; i < min((tid + 1) * CFACTOR, n_samples); ++i) {
            int bin_ibin_width = (int)((x[i] - x_min) / bin_width);
            bin_ibin_width = max(0, min(bin_ibin_width, bins - 1));
            atomicAdd(&bin_sums_s[bin_ibin_width], values[i]);
            atomicAdd(&bin_counts_s[bin_ibin_width], 1ULL);
        }
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 3: Merge block histograms into global histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            unsigned long long local_count = bin_counts_s[bin];
            double local_sum = bin_sums_s[bin];
            if (local_count > 0) {
                atomicAdd(&global_sums[bin], local_sum);
                atomicAdd(&global_counts[bin], local_count);
            }
        }
    }





/* Contiguous partitioning is inefficient on GPUs due to poor caching behavior.
 * Threads in a warp should access consecutive addresses to enable memory coalescing.
 * This motivates interleaved partitioning.
 * Below is a kernel using coarsening with interleaved partitioning.
 *
 * Each block:
 *  1. Initializes a shared-memory histogram (counts and sums)
 *  2. Processes samples using a grid-stride loop
 *  3. Accumulates into shared memory
 *  4. Atomically merges the block histogram into the global histogram
 */
__global__ void binned_statistic_kernel_v5(
        const int n_samples,
        const int bins,
        const double x_min,
        const double bin_width,
        const double * __restrict__ x,
        const double * __restrict__ values,
        unsigned long long * __restrict__ global_counts,
        double * __restrict__ global_sums
    )
    /* Parameters:
     *  n_samples     : Number of input samples
     *  bins        : Number of histogram bins
     *  x_min         : Lower bound of the histogram range
     *  bin_width     : Width of each histogram bin
     *  x             : Input coordinates (size n_samples)
     *  values        : Values to accumulate per bin (size n_samples)
     *  global_counts : Global histogram counts (size bins)
     *  global_sums   : Global accumulated values per bin (size bins)
     */
    {
        // Dynamically allocated shared memory (typed as double for alignment)
        extern __shared__ double shmem[];
        // Shared-memory views
        unsigned long long *bin_counts_s = reinterpret_cast<unsigned long long *>(shmem);
        double *bin_sums_s = reinterpret_cast<double *>(&bin_counts_s[bins]);
        /*------------------------------------------------------------
         * Phase 1: Initialize shared-memory histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            bin_counts_s[bin] = 0ULL;
            bin_sums_s[bin]   = 0.0;
        }
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 2: Grid-stride accumulation
         *------------------------------------------------------------
         */
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = tid; i <  n_samples; i+= blockDim.x * gridDim.x) {
            int bin_ibin_width = (int)((x[i] - x_min) / bin_width);
            bin_ibin_width = max(0, min(bin_ibin_width, bins - 1));
            atomicAdd(&bin_sums_s[bin_ibin_width], values[i]);
            atomicAdd(&bin_counts_s[bin_ibin_width], 1ULL);
        }
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 3: Merge block histograms into global histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            unsigned long long local_count = bin_counts_s[bin];
            double local_sum = bin_sums_s[bin];
            if (local_count > 0) {
                atomicAdd(&global_sums[bin],   local_sum);
                atomicAdd(&global_counts[bin], local_count);
            }
        }
    }





/* Third Idea: Aggregation
 *
 * When many identical values occur, atomic contention increases.
 * Aggregation lets each thread combine consecutive updates to the same
 * histogram bin into a single atomic operation, reducing contention.
 *
 * This adds some overhead and potential divergence, so it may be slower
 * when contention is low. However, under heavy contention, aggregation
 * can significantly improve performance.
 * Finally, note that, for many many bins, one may use shared memory tiling
 *
 * Each block:
 *  1. Initializes shared-memory histograms (counts and sums)
 *  2. Each thread processes samples in a grid-stride loop
 *     while locally accumulating consecutive hits to the same bin
 *  3. Flushes local accumulators into shared memory
 *  4. Atomically merges the block histogram into the global histogram
 */
    __global__ void binned_statistic_kernel_v6(
        const int n_samples,
        const int bins,
        const double x_min,
        const double bin_width,
        const double * __restrict__ x,
        const double * __restrict__ values,
        unsigned long long * __restrict__ global_counts,
        double * __restrict__ global_sums
    )
    /* Parameters:
     *  n_samples     : Number of input samples
     *  bins        : Number of histogram bins
     *  x_min         : Lower bound of the histogram range
     *  bin_width     : Width of each histogram bin
     *  x             : Input coordinates (size n_samples)
     *  values        : Values to accumulate per bin (size n_samples)
     *  global_counts : Global histogram counts (size bins)
     *  global_sums   : Global accumulated values per bin (size bins)
     */
    {
        // Dynamically allocated shared memory (typed as double for alignment)
        extern __shared__ double shmem[];
        // Shared-memory views
        unsigned long long *bin_counts_s = reinterpret_cast<unsigned long long *>(shmem);
        double *bin_sums_s = reinterpret_cast<double *>(&bin_counts_s[bins]);
        /*------------------------------------------------------------
         * Phase 1: Initialize shared-memory histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            bin_counts_s[bin] = 0ULL;
            bin_sums_s[bin]   = 0.0;
        }
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 2: Grid-stride accumulation with run-length optimization
         *------------------------------------------------------------
         */
        unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride     = blockDim.x * gridDim.x;
        int prev_bin_ibin_width = -1;
        unsigned long long local_count = 0ULL;
        double local_sum = 0.0;
        for (unsigned int i = global_tid; i < n_samples; i += stride) {
            int bin_ibin_width = (int)((x[i] - x_min) / bin_width);
            bin_ibin_width = max(0, min(bin_ibin_width, bins - 1));
            if (bin_ibin_width == prev_bin_ibin_width) {
                // Accumulate consecutive hits to the same bin
                local_count += 1ULL;
                local_sum   += values[i];
            } else {
                // Flush previous bin accumulator
                if (local_count > 0) {
                    atomicAdd(&bin_counts_s[prev_bin_ibin_width], local_count);
                    atomicAdd(&bin_sums_s[prev_bin_ibin_width],   local_sum);
                }
                // Start new accumulator
                prev_bin_ibin_width = bin_ibin_width;
                local_count  = 1ULL;
                local_sum    = values[i];
            }
        }
        // Flush final accumulator
        if (local_count > 0) {
            atomicAdd(&bin_counts_s[prev_bin_ibin_width], local_count);
            atomicAdd(&bin_sums_s[prev_bin_ibin_width], local_sum);
        }
        __syncthreads();
        /*------------------------------------------------------------
         * Phase 3: Merge block histograms into global histograms
         *------------------------------------------------------------
         */
        for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
            unsigned long long block_count = bin_counts_s[bin];
            double block_sum = bin_sums_s[bin];
            if (block_count > 0ULL) {
                atomicAdd(&global_sums[bin],   block_sum);
                atomicAdd(&global_counts[bin], block_count);
            }
        }
    }





/*
 * CUDA kernel computing the per-bin mean from accumulated sums
 * and histogram counts.
 */
    __global__ void binned_statistic_mean(
        const int bins,
        const unsigned long long * __restrict__ bin_counts,
        const double * __restrict__ bin_sums,
        double * __restrict__ bin_means
    )
    /* Parameters:
     *  bins        : Number of histogram bins
     *  bin_counts    : Histogram counts per bin (size bins)
     *  bin_sums      : Accumulated values per bin (size bins)
     *  bin_means     : Output mean per bin (size bins)
     */
    {
        // Global thread index
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        // Grid-stride loop over bins
        for (unsigned int bin = tid; bin < bins; bin += stride) {
            unsigned long long count = bin_counts[bin];
            // Avoid division by zero
            bin_means[bin] = (count > 0ULL) ? (bin_sums[bin] / static_cast<double>(count)) : 0.0;
        }
    }

// For many many bins, one may use shared memory tiling
"""

bin_stat_mod = cp.RawModule(
    code=bin_stat_code,
    backend="nvrtc",
    options=("-std=c++17",),
    name_expressions=[
        "binned_statistic_kernel_v1",
        "binned_statistic_kernel_v2",
        "binned_statistic_kernel_v3",
        "binned_statistic_kernel_v4",
        "binned_statistic_kernel_v5",
        "binned_statistic_kernel_v6",
        "binned_statistic_mean",
    ],
)


def binned_statistic_v1(x_gpu, values_gpu, bins, n_threads=256):
    """
    Compute a binned statistic (mean) on the GPU using custom CUDA kernels.

    Parameters
    ----------
    x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    n_samples = len(x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    x_min = cp.min(x_gpu).get().item()
    x_max = cp.max(x_gpu).get().item()
    bin_width = (x_max - x_min) / bins
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v1")
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min,
            bin_width,
            x_gpu,
            values_gpu,
            hist_count,
            hist_value,
        ),
    )
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v2(x_gpu, values_gpu, bins, n_threads=256):
    """
    Compute a binned statistic (mean) on the GPU using custom CUDA kernels.

    Parameters
    ----------
    x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    n_samples = len(x_gpu)
    statistic = cp.zeros(bins, dtype=cp.float64)
    x_min = cp.min(x_gpu).get().item()
    x_max = cp.max(x_gpu).get().item()
    bin_width = (x_max - x_min) / bins
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v2")
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    hist_value = cp.zeros(bins * n_blocks, dtype=cp.float64)
    hist_count = cp.zeros(bins * n_blocks, dtype=cp.int64)
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min,
            bin_width,
            x_gpu,
            values_gpu,
            hist_count,
            hist_value,
        ),
    )
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v3(x_gpu, values_gpu, bins, max_shared_mem=None, n_threads=256):
    """
    Compute a binned statistic (mean) on the GPU using custom CUDA kernels.

    Parameters
    ----------
    x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    n_samples = len(x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    x_min = cp.min(x_gpu).get().item()
    x_max = cp.max(x_gpu).get().item()
    bin_width = (x_max - x_min) / bins
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v3")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min,
            bin_width,
            x_gpu,
            values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v4(x_gpu, values_gpu, bins, max_shared_mem=None, n_threads=256):
    """
    Compute a binned statistic (mean) on the GPU using custom CUDA kernels.

    Parameters
    ----------
    x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    n_samples = len(x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    x_min = cp.min(x_gpu).get().item()
    x_max = cp.max(x_gpu).get().item()
    bin_width = (x_max - x_min) / bins
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v4")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min,
            bin_width,
            x_gpu,
            values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v5(x_gpu, values_gpu, bins, max_shared_mem=None, n_threads=256):
    """
    Compute a binned statistic (mean) on the GPU using custom CUDA kernels.

    Parameters
    ----------
    x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    n_samples = len(x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    x_min = cp.min(x_gpu).get().item()
    x_max = cp.max(x_gpu).get().item()
    bin_width = (x_max - x_min) / bins
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v5")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min,
            bin_width,
            x_gpu,
            values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v6(x_gpu, values_gpu, bins, max_shared_mem=None, n_threads=256):
    """
    Compute a binned statistic (mean) on the GPU using custom CUDA kernels.

    Parameters
    ----------
    x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    n_samples = len(x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    x_min = cp.min(x_gpu).get().item()
    x_max = cp.max(x_gpu).get().item()
    bin_width = (x_max - x_min) / bins
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v6")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min,
            bin_width,
            x_gpu,
            values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def scatter_v0(comm, array_gpu):
    """
    Scatter a 1D GPU array across all ranks of a communicator.

    Each rank receives a contiguous chunk of the input array.
    If the array size is not divisible by the number of ranks,
    the first ranks receive one extra element.

    Parameters
    ----------
    comm : communicator
        Distributed communicator object (MPI-like).
    array_gpu : cupy.ndarray
        Input array on the GPU (only significant on rank 0).

    Returns
    -------
    local_array_gpu : cupy.ndarray
        Local portion of the array on each rank (GPU).
    """
    # Get communicator info
    rank = comm._comm.rank_id()
    world_size = comm._comm.size()
    # Define local array
    n = len(array_gpu)
    loc_n, rem_n = n // world_size, n % world_size
    loc_ns = np.full(world_size, loc_n, dtype=np.int64)
    loc_ns[:rem_n] += 1
    max_loc_n = np.max(loc_ns)
    cumsum_loc_n = np.zeros(world_size + 1, dtype=np.int64)
    np.cumsum(loc_ns, out=cumsum_loc_n[1:], dtype=np.int64)
    local_array_gpu = cp.zeros(loc_ns[rank])
    # Define communication buffers
    sendbuf = cp.zeros((world_size, max_loc_n))
    recvbuf = cp.zeros(max_loc_n)
    if rank == 0:
        for i in range(world_size):
            sendbuf[i, : loc_ns[i]] = array_gpu[cumsum_loc_n[i]: cumsum_loc_n[i + 1]]
    comm.scatter(sendbuf, recvbuf, root=0)
    local_array_gpu[:] = recvbuf[: loc_ns[rank]]
    return local_array_gpu


def scatter_v1(comm, array_gpu):
    """
    Scatter a 1D GPU array across all ranks of a communicator.

    Each rank receives a contiguous chunk of the input array.
    If the array size is not divisible by the number of ranks,
    the first ranks receive one extra element.

    Parameters
    ----------
    comm : communicator
        Distributed communicator object (MPI-like).
    array_gpu : cupy.ndarray
        Input array on the GPU (only significant on rank 0).

    Returns
    -------
    local_array_gpu : cupy.ndarray
        Local portion of the array on each rank (GPU).
    """
    # Get rank and world size
    rank = comm._comm.rank_id()
    world_size = comm._comm.size()
    # Total number of elements
    n = len(array_gpu)
    # Compute local sizes
    loc_n, rem_n = n // world_size, n % world_size
    loc_ns = np.full(world_size, loc_n, dtype=np.int64)
    loc_ns[:rem_n] += 1
    # Compute displacements
    cumsum_loc_n = np.zeros(world_size + 1, dtype=np.int64)
    np.cumsum(loc_ns, out=cumsum_loc_n[1:], dtype=np.int64)
    # Allocate local array (exact size)
    local_array_gpu = cp.zeros(loc_ns[rank], dtype=array_gpu.dtype)
    if rank == 0:
        # Root sends slices directly to each rank
        local_array_gpu[:] = array_gpu[cumsum_loc_n[0]: cumsum_loc_n[1]]
        for i in range(1, world_size):
            # Send exact slice (no padding)
            comm.send(array_gpu[cumsum_loc_n[i]: cumsum_loc_n[i + 1]], i)
    else:
        # Non-root ranks receive exactly their slice
        comm.recv(local_array_gpu, 0)
    return local_array_gpu


def binned_statistic_v1_dist(
    comm, local_x_gpu, local_values_gpu, bins, n_threads=256
):
    """
    Compute a binned statistic (mean) on GPUs using custom CUDA kernels for distributed data.

    Parameters
    ----------
    local_x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    local_values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    # Get minimum and maximum accross all the ranks
    # This is necessary for nccl cupy otherwise error
    local_x_min = cp.min(local_x_gpu) * cp.ones(2)
    # It needs an array, does not work on single number
    local_x_max = cp.max(local_x_gpu) * cp.ones(2)
    x_min = cp.zeros(2)
    x_max = cp.zeros(2)
    comm.all_reduce(local_x_min, x_min, op="min")
    comm.all_reduce(local_x_max, x_max, op="max")
    x_min_v = x_min[0].get().item()
    x_max_v = x_max[0].get().item()
    bin_width = (x_max_v - x_min_v) / bins
    # Initialize all required arrays and kernels
    n_samples = len(local_x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v1")
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    # Call  binned_statistic_kernel
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min_v,
            bin_width,
            local_x_gpu,
            local_values_gpu,
            hist_count,
            hist_value,
        ),
    )
    # Reduce the histogram with the value and the count summing accross all the ranks
    comm.all_reduce(hist_value, hist_value)
    comm.all_reduce(hist_count, hist_count)
    # Compute the mean per bin
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v2_dist(
    comm, local_x_gpu, local_values_gpu, bins, n_threads=256
):
    """
    Compute a binned statistic (mean) on GPUs using custom CUDA kernels for distributed data.

    Parameters
    ----------
    local_x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    local_values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    # Get minimum and maximum accross all the ranks
    # This is necessary for nccl cupy otherwise error
    local_x_min = cp.min(local_x_gpu) * cp.ones(2)
    # It needs an array, does not work on single number
    local_x_max = cp.max(local_x_gpu) * cp.ones(2)
    x_min = cp.zeros(2)
    x_max = cp.zeros(2)
    comm.all_reduce(local_x_min, x_min, op="min")
    comm.all_reduce(local_x_max, x_max, op="max")
    x_min_v = x_min[0].get().item()
    x_max_v = x_max[0].get().item()
    bin_width = (x_max_v - x_min_v) / bins
    # Initialize all required arrays and kernels
    n_samples = len(local_x_gpu)
    statistic = cp.zeros(bins, dtype=cp.float64)
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v2")
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    hist_value = cp.zeros(bins * n_blocks, dtype=cp.float64)
    hist_count = cp.zeros(bins * n_blocks, dtype=cp.int64)
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min_v,
            bin_width,
            local_x_gpu,
            local_values_gpu,
            hist_count,
            hist_value,
        ),
    )
    # Reduce the histogram with the value and the count summing accross all the ranks
    comm.all_reduce(hist_value, hist_value)
    comm.all_reduce(hist_count, hist_count)
    # Compute the mean per bin
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v3_dist(
    comm, local_x_gpu, local_values_gpu, bins, max_shared_mem=None, n_threads=256
):
    """
    Compute a binned statistic (mean) on GPUs using custom CUDA kernels for distributed data.

    Parameters
    ----------
    local_x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    local_values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    # Get minimum and maximum accross all the ranks
    # This is necessary for nccl cupy otherwise error
    local_x_min = cp.min(local_x_gpu) * cp.ones(2)
    # It needs an array, does not work on single number
    local_x_max = cp.max(local_x_gpu) * cp.ones(2)
    x_min = cp.zeros(2)
    x_max = cp.zeros(2)
    comm.all_reduce(local_x_min, x_min, op="min")
    comm.all_reduce(local_x_max, x_max, op="max")
    x_min_v = x_min[0].get().item()
    x_max_v = x_max[0].get().item()
    bin_width = (x_max_v - x_min_v) / bins
    # Initialize all required arrays and kernels
    n_samples = len(local_x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v3")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    # The kernels needs shared memory as two array (2) of double (8) each has size the number of bins (bins)
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min_v,
            bin_width,
            local_x_gpu,
            local_values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    # Reduce the histogram with the value and the count summing accross all the ranks
    comm.all_reduce(hist_value, hist_value)
    comm.all_reduce(hist_count, hist_count)
    # Compute the mean per bin
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v4_dist(
    comm, local_x_gpu, local_values_gpu, bins, max_shared_mem=None, n_threads=256
):
    """
    Compute a binned statistic (mean) on GPUs using custom CUDA kernels for distributed data.

    Parameters
    ----------
    local_x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    local_values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    # Get minimum and maximum accross all the ranks
    # This is necessary for nccl cupy otherwise error
    local_x_min = cp.min(local_x_gpu) * cp.ones(2)
    # It needs an array, does not work on single number
    local_x_max = cp.max(local_x_gpu) * cp.ones(2)
    x_min = cp.zeros(2)
    x_max = cp.zeros(2)
    comm.all_reduce(local_x_min, x_min, op="min")
    comm.all_reduce(local_x_max, x_max, op="max")
    x_min_v = x_min[0].get().item()
    x_max_v = x_max[0].get().item()
    bin_width = (x_max_v - x_min_v) / bins
    # Initialize all required arrays and kernels
    n_samples = len(local_x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v4")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min_v,
            bin_width,
            local_x_gpu,
            local_values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    # Reduce the histogram with the value and the count summing accross all the ranks
    comm.all_reduce(hist_value, hist_value)
    comm.all_reduce(hist_count, hist_count)
    # Compute the mean per bin
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v5_dist(
    comm, local_x_gpu, local_values_gpu, bins, max_shared_mem=None, n_threads=256
):
    """
    Compute a binned statistic (mean) on GPUs using custom CUDA kernels for distributed data.

    Parameters
    ----------
    local_x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    local_values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    # Get minimum and maximum accross all the ranks
    # This is necessary for nccl cupy otherwise error
    local_x_min = cp.min(local_x_gpu) * cp.ones(2)
    # It needs an array, does not work on single number
    local_x_max = cp.max(local_x_gpu) * cp.ones(2)
    x_min = cp.zeros(2)
    x_max = cp.zeros(2)
    comm.all_reduce(local_x_min, x_min, op="min")
    comm.all_reduce(local_x_max, x_max, op="max")
    x_min_v = x_min[0].get().item()
    x_max_v = x_max[0].get().item()
    bin_width = (x_max_v - x_min_v) / bins
    # Initialize all required arrays and kernels
    n_samples = len(local_x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v5")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min_v,
            bin_width,
            local_x_gpu,
            local_values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    # Reduce the histogram with the value and the count summing accross all the ranks
    comm.all_reduce(hist_value, hist_value)
    comm.all_reduce(hist_count, hist_count)
    # Compute the mean per bin
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


def binned_statistic_v6_dist(
    comm, local_x_gpu, local_values_gpu, bins, max_shared_mem=None, n_threads=256
):
    """
    Compute a binned statistic (mean) on GPUs using custom CUDA kernels for distributed data.

    Parameters
    ----------
    local_x_gpu : cupy.ndarray
        A sequence of values to be binned (on GPU).
    local_values_gpu : cupy.ndarray
        The data on which the statistic will be computed (on GPU).
        This must be the same shape as x, or a set of sequences - each the same shape as x.
    bins : int
        Number of bins.
    max_shared_mem: int, optional
        The maximum size in bytes of dynamic shared memory per block for this function.
        Any launch must have a dynamic shared memory size smaller than this value.
    n_threads : int, optional
        Number of CUDA threads per block (default: 256).

    Returns
    -------
    statistic : cupy.ndarray
        Mean value per bin.
    """
    # Get minimum and maximum accross all the ranks
    # This is necessary for nccl cupy otherwise error
    local_x_min = cp.min(local_x_gpu) * cp.ones(2)
    # It needs an array, does not work on single number
    local_x_max = cp.max(local_x_gpu) * cp.ones(2)
    x_min = cp.zeros(2)
    x_max = cp.zeros(2)
    comm.all_reduce(local_x_min, x_min, op="min")
    comm.all_reduce(local_x_max, x_max, op="max")
    x_min_v = x_min[0].get().item()
    x_max_v = x_max[0].get().item()
    bin_width = (x_max_v - x_min_v) / bins
    # Initialize all required arrays and kernels
    n_samples = len(local_x_gpu)
    hist_value = cp.zeros(bins, dtype=cp.float64)
    hist_count = cp.zeros(bins, dtype=cp.int64)
    statistic = cp.zeros(bins, dtype=cp.float64)
    binned_statistic_kernel = bin_stat_mod.get_function("binned_statistic_kernel_v6")
    if max_shared_mem is not None:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = max_shared_mem
    else:
        binned_statistic_kernel.max_dynamic_shared_size_bytes = 2 * 8 * bins
    binned_statistic_mean = bin_stat_mod.get_function("binned_statistic_mean")
    n_blocks = (n_samples // n_threads) + 1
    binned_statistic_kernel(
        (n_blocks,),
        (n_threads,),
        (
            n_samples,
            bins,
            x_min_v,
            bin_width,
            local_x_gpu,
            local_values_gpu,
            hist_count,
            hist_value,
        ),
        shared_mem=(2 * 8 * bins),
    )
    # Reduce the histogram with the value and the count summing accross all the ranks
    comm.all_reduce(hist_value, hist_value)
    comm.all_reduce(hist_count, hist_count)
    # Compute the mean per bin
    n_blocks_mean = (bins // n_threads) + 1
    binned_statistic_mean(
        (n_blocks_mean,), (n_threads,), (bins, hist_count, hist_value, statistic)
    )
    return statistic


scatter = scatter_v0

binned_statistic_dist = binned_statistic_v6_dist

binned_statistic = binned_statistic_v6
