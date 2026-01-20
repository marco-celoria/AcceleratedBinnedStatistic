# Accelerated Binned Statistic

Accelerated Binned Statistic is an Python package for a simple porting to multi-GPUs of the `scipy.stats.binned_statistic` function.

The function is a generalization of a histogram function. 

While a simple histogram divides the space into bins, and returns the count of the number of points in each bin, the binned statistic function allows the computation of the mean of the values (or set of values) within each bin.

Compared to the `scipy.stats.binned_statistic`, so far it works only with the `statistic='mean'` and returning only the `statistic` array (i.e. the values of the selected statistic in each bin).

It is based on `CuPy` and its Distributed extension for the NCCL communication, as well as custom CUDA kernels.

The code has been tested on Leonardo booster partition at Cineca, and was developed during a Epicure Hackathon.

While having less options than the `scipy.stats.binned_statistic`, the Accelerated Binned Statistic is orders of magnitude faster for large workloads and able to scale on several GPUs. 

# Installation

You can install the current version from GitHub and follow the following scripts

- `submit_install.sh` (using pip)
- `submit_install_uv.sh` (using uv)

# Examples

This is a basic example which shows you how to use the function on a single GPU

```
n_samples = 40000
n_bins = 15
x_cpu = np.sin(np.linspace(0.0, 4.0, n_samples))
values_cpu = np.cos(np.linspace(0.0, 4.0, n_samples))
statistic_cpu, _, _ = scipy.stats.binned_statistic(x_cpu, values_cpu, bins=n_bins)

x_gpu = cp.asarray(x_cpu, dtype=cp.float64)
values_gpu = cp.asarray(values_cpu, dtype=cp.float64)
statistic_gpu = acc.binned_statistic(x_gpu, values_gpu, n_bins)
```

For multi-GPUs

```
comm = init_process_group(world_size, rank, use_mpi=True)

n_samples = 40000
n_bins = 15

arr_cpu = np.linspace(0.0, 4.0, n_samples)
x_cpu = 0.01 * np.sin(arr_cpu) + arr_cpu**2 + 1.0
values_cpu = arr_cpu + 0.01 * np.cos(arr_cpu) + 2.0
statistic_cpu, _, _ = scipy.stats.binned_statistic(x_cpu, values_cpu, bins=n_bins)

arr_gpu = cp.asarray(arr_cpu, dtype=cp.float64)
local_arr_gpu = acc.scatter(comm, arr_gpu)
local_x_gpu = 0.01 * np.sin(local_arr_gpu) + local_arr_gpu**2 + 1.0
local_values_gpu = local_arr_gpu + 0.01 * np.cos(local_arr_gpu) + 2.0

statistic_gpu = acc.binned_statistic_dist(comm, local_x_gpu, local_values_gpu, bins=n_bins)
```

see the files in `examples` for more details.

# Benchmarks

We have considered 6 implementations of the CUDA kernels, depending on the parallelization startegy:

- A) Basic implementation
- B) Privatization in Global Memory
- C) Privatization in Shared Memory
- D) Coarsening with contiguous partitioning
- E) Coarsening with interleaved partitioning 
- F) Aggregation

Preliminary benchmarks on Leonardo booster partition (single socket 32-core Intel Xeon Platinum 8358 CPU, 2.60GHz and 4x NVIDIA custom Ampere A100 GPU 64GB HBM2e per node) suggests that for optimized kernels we have:

```
Samples     = 413340001
Bins        = 6999 
CPU    time = 16.938 s 
1 GPU  time =  0.285 s
4 GPUs time =  0.071 s
```

 

