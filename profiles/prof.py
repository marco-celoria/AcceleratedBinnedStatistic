# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import cupy as cp
from cupyx.distributed import init_process_group
import os
import time
import gc
import acceleratedbinnedstatistic.binned_statistic as acc

def bench(comm, x_cpu, values_cpu, n_bins, max_shared_mem = None, n_threads = 256):
    rank = comm._comm.rank_id()
    world_size = comm._comm.size()
    if rank == 0:
        x_gpu = cp.asarray(x_cpu, dtype=cp.float64)
        values_gpu = cp.asarray(values_cpu, dtype=cp.float64)
    else:
        x_gpu = cp.zeros(x_cpu.shape, dtype=cp.float64)
        values_gpu = cp.zeros(values_cpu.shape, dtype=cp.float64)
    local_x_gpu = acc.scatter(comm, x_gpu)
    local_values_gpu = acc.scatter(comm, values_gpu)
    cp.cuda.get_current_stream().synchronize()
    statistic_v1_dist = acc.binned_statistic_v1_dist(comm, local_x_gpu, local_values_gpu, n_bins, n_threads)
    cp.cuda.get_current_stream().synchronize()
    statistic_v2_dist = acc.binned_statistic_v2_dist(comm, local_x_gpu, local_values_gpu, n_bins, n_threads)
    cp.cuda.get_current_stream().synchronize()
    statistic_v3_dist = acc.binned_statistic_v3_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    cp.cuda.get_current_stream().synchronize()
    statistic_v4_dist = acc.binned_statistic_v4_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    cp.cuda.get_current_stream().synchronize()
    statistic_v5_dist = acc.binned_statistic_v5_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    cp.cuda.get_current_stream().synchronize()
    statistic_v6_dist = acc.binned_statistic_v6_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    cp.cuda.get_current_stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

def main():
    
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    #print(f"world_size = {world_size}", flush=True)
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    #print(f"rank = {rank}", flush=True)
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    #print(f"local_rank = {local_rank}", flush=True)
    dev = cp.cuda.Device(local_rank)
    dev.use()
    comm = init_process_group(world_size, rank, use_mpi=True)
    n_samples = 41334
    if rank == 0:
        x_cpu = 0.01 * np.sin(np.linspace(0., 5., n_samples)) + np.linspace(0., 10., n_samples)**2 + 1.
        values_cpu = np.linspace(0., 10., n_samples) + 0.01 * np.cos(np.linspace(0., 5., n_samples)) + 2.
    else:
        x_cpu = np.zeros(n_samples)
        values_cpu = np.zeros(n_samples)
    n_bins = 1533
    sh_mems = 40000
    n_threads = 256
    bench(comm,  x_cpu, values_cpu, n_bins, sh_mems, n_threads)


if __name__ == "__main__":
    main()


