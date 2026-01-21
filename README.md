# Accelerated Binned Statistic

Accelerated Binned Statistic is an Python package for a simple porting to multi-GPUs of the `scipy.stats.binned_statistic` function.

The function is a generalization of a histogram function. 

While a simple histogram divides the space into bins, and returns the count of the number of points in each bin, the binned statistic function allows the computation of the mean of the values (or set of values) within each bin.

Compared to the `scipy.stats.binned_statistic`, so far it works only with the `statistic='mean'` and returning only the `statistic` array (i.e. the values of the selected statistic in each bin).

It is based on `CuPy` and its Distributed extension for the NCCL communication, as well as custom CUDA kernels.

The code has been tested on Leonardo booster partition at Cineca, and was developed during a Epicure Hackathon.

While having less options than the `scipy.stats.binned_statistic`, the Accelerated Binned Statistic is orders of magnitude faster for large workloads and able to scale on several GPUs. 

# Installation

The code has been tested on Leonardo at Cineca with the following software stack
- CUDA 12.1 
- GCC 12.2.0 
- Python 3.11.6
- OpenMPI 4.1.6

Specifically on Leonardo by loading the following modules

```
module load cuda/12.1
module load gcc/12.2.0-binut2.41
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.6--gcc--8.5.0
```

the other dependencies can be installed using `pip` or `uv` and are described in `requirements.txt` or `pyproject.toml`.  

As an example, on Leonardo you can install the current version from GitHub using one of the scripts below:

- `submit_install.sh`    (using pip)
- `submit_install_uv.sh` (using uv)

For other systems, you can follow a similar strategy.

# Examples

This is a basic example which shows you how to use the function on a single GPU

```
n_samples = 40000
n_bins = 15
arr_cpu = np.linspace(0.0, 4.0, n_samples)

# With scipy we do as usual
x_cpu = 0.01 * np.sin(arr_cpu) + arr_cpu**2 + 1.0
v_cpu = 0.01 * np.cos(arr_cpu) + arr_cpu + 2.0
statistic_cpu, _, _ = scipy.stats.binned_statistic(x_cpu, v_cpu, bins=n_bins)

# To use the accelerated function, first move data on device with CuPy
x_gpu = cp.asarray(x_cpu, dtype=cp.float64)
v_gpu = cp.asarray(v_cpu, dtype=cp.float64)
# Then call the accelerated function
statistic_gpu = acc.binned_statistic(x_gpu, v_gpu, n_bins)

# To call a specific version among the possible CUDA kernel implementations
statistic_gpu_v2 = acc.binned_statistic_v2(x_gpu, v_gpu, n_bins)
```

For multi-GPUs

```
# Set up the NCCLBackend communications
# ...
comm = init_process_group(world_size, rank, use_mpi=True)

n_samples = 40000
n_bins = 15
# Suppose only rank 0 initializes the arrays 
if rank == 0:
    arr_cpu = np.linspace(0.0, 4.0, n_samples)
    x_cpu = 0.01 * np.sin(arr_cpu) + arr_cpu**2 + 1.0
    v_cpu = 0.01 * np.cos(arr_cpu) + arr_cpu + 2.0
    statistic_cpu, _, _ = scipy.stats.binned_statistic(x_cpu, v_cpu, bins=n_bins)
else: # While the others ranks just allocate memory 
    arr_cpu = np.zeros(n_samples)
# First we need to move data on device    
arr_gpu = cp.asarray(arr_cpu, dtype=cp.float64)
# as we want to scatter the data from rank 0 to all the other ranks
# and acc.scatter works using NCCL requiring data to be already on GPUs
local_arr_gpu = acc.scatter(comm, arr_gpu)

# Once we have the local shards, we can compute the accelerated binned_statistic 
local_x_gpu = 0.01 * np.sin(local_arr_gpu) + local_arr_gpu**2 + 1.0
local_v_gpu = 0.01 * np.cos(local_arr_gpu) + local_arr_gpu + 2.0
statistic_gpu = acc.binned_statistic_dist(comm, local_x_gpu, local_v_gpu, n_bins)

# To call a specific version among the possible CUDA kernel implementations
statistic_gpu_v2 = acc.binned_statistic_v2_dist(comm, local_x_gpu, local_v_gpu, n_bins)
```

see the files in `examples` for more details.

# Benchmarks

We have considered 6 implementations of the CUDA kernels, depending on the parallelization startegy:

- v1) Basic implementation
- v2) Privatization in Global Memory
- v3) Privatization in Shared Memory
- v4) Coarsening with contiguous partitioning
- v5) Coarsening with interleaved partitioning 
- v6) Aggregation

Preliminary benchmarks on Leonardo booster partition (single socket 32-core Intel Xeon Platinum 8358 CPU, 2.60GHz and 4x NVIDIA custom Ampere A100 GPU 64GB HBM2e per node) suggests that for input array of size 413340001 (sample size), we have

| Sample size | Bins | `scipy.stats.binned_statistic` on CPU [s] |
|-------------| -----|--------------------------------------------|
| 413340001   | 433  | 22.2393                                    |
| 413340001   | 833  | 15.7343                                    |
| 413340001   | 1533 | 16.0404                                    |
| 413340001   | 2999 | 16.4502                                    |
| 413340001   | 6999 | 16.8233                                    |

| Sample size | Bins | `v1` on 1GPU [s] | `v2` on 1GPU [s] | `v3` on 1GPU [s] | `v4` on 1GPU [s] | `v5` on 1GPU [s] | `v6` on 1GPU [s] | threads x block |
|-------------|------|------------------|------------------|------------------|------------------|------------------|------------------|-------------------|
| 413340001   | 433  | 1.088988         | 0.052489         | 0.047707         | 0.045374         | 0.047586         | 0.047448         | 128               | 
| 413340001   | 833  | 0.907774         | 0.052030         | 0.068222         | 0.063465         | 0.068295         | 0.068371         | 256               |
| 413340001   | 1533 | 0.886425         | 0.050479         | 0.096039         | 0.090732         | 0.098262         | 0.096747         | 512               |
| 413340001   | 2999 | 0.857417         | 0.059253         | 0.160423         | 0.136270         | 0.161757         | 0.160999         | 768               |
| 413340001   | 6999 | 0.795643         | 0.076116         | 0.289973         | 0.240219         | 0.290002         | 0.285196         | 1024              |

| Sample size | Bins | `v1` on 4GPU [s] | `v2` on 4GPU [s] | `v3` on 4GPU [s] | `v4` on 4GPU [s] | `v5` on 4GPU [s] | `v6` on 4GPU [s] | threads x block |
|-------------|------|------------------|------------------|------------------|------------------|------------------|------------------|-------------------|
| 413340001   | 433  | 0.278728         | 0.051490         | 0.012159         | 0.011569         | 0.012102         | 0.012062         | 128               | 
| 413340001   | 833  | 0.233641         | 0.050243         | 0.017291         | 0.016082         | 0.017293         | 0.017306         | 256               |
| 413340001   | 1533 | 0.233319         | 0.046970         | 0.024268         | 0.022972         | 0.024813         | 0.024427         | 512               |
| 413340001   | 2999 | 0.230651         | 0.059358         | 0.040440         | 0.034340         | 0.040756         | 0.040556         | 768               |
| 413340001   | 6999 | 0.220536         | 0.096494         | 0.072972         | 0.060662         | 0.072985         | 0.071776         | 1024              |
 
Note that v2 is very demanding in terms of global memory (`2 * n_bins * n_blocks * 8` bytes for storing intermediate results, where `n_blocks = sample_size / n_threads`) whereas v3, v4, v5, v6 require `2 * 8 * n_bins` bytes of shared memory.
