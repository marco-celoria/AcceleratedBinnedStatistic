# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import cupy as cp
import numpy as np
import scipy
from cupyx.distributed import init_process_group

import acceleratedbinnedstatistic.binned_statistic as acc


def main():

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID")))
    dev = cp.cuda.Device(local_rank)
    dev.use()
    comm = init_process_group(world_size, rank, use_mpi=True)

    n_samples = 40000
    arr_cpu = np.linspace(0.0, 4.0, n_samples)
    arr_gpu = cp.asarray(arr_cpu, dtype=cp.float64)
    local_arr_gpu = acc.scatter(comm, arr_gpu)

    # print(f"On rank {rank}):\n"
    #        f"Head of local_arr_gpu:{local_arr_gpu[:5]}\n"
    #        f"Tail of local_arr_gpu:{local_arr_gpu[-5:]}", flush=True)

    x_cpu = 0.01 * np.sin(arr_cpu) + arr_cpu**2 + 1.0
    values_cpu = arr_cpu + 0.01 * np.cos(arr_cpu) + 2.0

    local_x_gpu = 0.01 * np.sin(local_arr_gpu) + local_arr_gpu**2 + 1.0
    local_values_gpu = local_arr_gpu + 0.01 * np.cos(local_arr_gpu) + 2.0

    n_bins = 15
    statistic_gpu = acc.binned_statistic_dist(
        comm, local_x_gpu, local_values_gpu, n_bins
    )
    print(
        f"On rank {rank}) Results from acceleratedbinnedstatistic\n{statistic_gpu}",
        flush=True,
    )
    statistic_cpu, _, _ = scipy.stats.binned_statistic(x_cpu, values_cpu, bins=n_bins)
    if rank == 0:
        print(f"Results from scipy:\n{statistic_cpu}", flush=True)
    assert np.allclose(statistic_cpu, statistic_gpu.get())


if __name__ == "__main__":
    main()
