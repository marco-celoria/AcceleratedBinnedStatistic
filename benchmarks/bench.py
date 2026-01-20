# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import gc
import os

import cupy as cp
import numpy as np
from cupyx.distributed import init_process_group
from cupyx.profiler import benchmark

import acceleratedbinnedstatistic.binned_statistic as acc

n_iterations = 10


def bench(comm, x_cpu, values_cpu, n_bins, max_shared_mem=None, n_threads=256):
    n_samples = x_cpu.shape[0]
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
    start_gpu_1 = cp.cuda.Event()
    start_gpu_2 = cp.cuda.Event()
    start_gpu_3 = cp.cuda.Event()
    start_gpu_4 = cp.cuda.Event()
    start_gpu_5 = cp.cuda.Event()
    start_gpu_6 = cp.cuda.Event()
    end_gpu_1 = cp.cuda.Event()
    end_gpu_2 = cp.cuda.Event()
    end_gpu_3 = cp.cuda.Event()
    end_gpu_4 = cp.cuda.Event()
    end_gpu_5 = cp.cuda.Event()
    end_gpu_6 = cp.cuda.Event()
    if rank == 0:
        cp.cuda.get_current_stream().synchronize()
        start_gpu_1.record()
        for _ in range(n_iterations):
            acc.binned_statistic_v1(x_gpu, values_gpu, n_bins, n_threads)
        cp.cuda.get_current_stream().synchronize()
        end_gpu_1.record()
        end_gpu_1.synchronize()
        t1_gpu = (
            (cp.cuda.get_elapsed_time(start_gpu_1, end_gpu_1)) / (n_iterations) * 1e-3
        )

        cp.cuda.get_current_stream().synchronize()
        start_gpu_2.record()
        for _ in range(n_iterations):
            acc.binned_statistic_v2(x_gpu, values_gpu, n_bins, n_threads)
        cp.cuda.get_current_stream().synchronize()
        end_gpu_2.record()
        end_gpu_2.synchronize()
        t2_gpu = (
            (cp.cuda.get_elapsed_time(start_gpu_2, end_gpu_2)) / (n_iterations) * 1e-3
        )

        cp.cuda.get_current_stream().synchronize()
        start_gpu_3.record()
        for _ in range(n_iterations):
            acc.binned_statistic_v3(
                x_gpu, values_gpu, n_bins, max_shared_mem, n_threads
            )
        cp.cuda.get_current_stream().synchronize()
        end_gpu_3.record()
        end_gpu_3.synchronize()
        t3_gpu = (
            (cp.cuda.get_elapsed_time(start_gpu_3, end_gpu_3)) / (n_iterations) * 1e-3
        )

        cp.cuda.get_current_stream().synchronize()
        start_gpu_4.record()
        for _ in range(n_iterations):
            acc.binned_statistic_v4(
                x_gpu, values_gpu, n_bins, max_shared_mem, n_threads
            )
        cp.cuda.get_current_stream().synchronize()
        end_gpu_4.record()
        end_gpu_4.synchronize()
        t4_gpu = (
            (cp.cuda.get_elapsed_time(start_gpu_4, end_gpu_4)) / (n_iterations) * 1e-3
        )

        cp.cuda.get_current_stream().synchronize()
        start_gpu_5.record()
        for _ in range(n_iterations):
            acc.binned_statistic_v5(
                x_gpu, values_gpu, n_bins, max_shared_mem, n_threads
            )
        cp.cuda.get_current_stream().synchronize()
        end_gpu_5.record()
        end_gpu_5.synchronize()
        t5_gpu = (
            (cp.cuda.get_elapsed_time(start_gpu_5, end_gpu_5)) / (n_iterations) * 1e-3
        )

        cp.cuda.get_current_stream().synchronize()
        start_gpu_6.record()
        for _ in range(n_iterations):
            acc.binned_statistic_v6(
                x_gpu, values_gpu, n_bins, max_shared_mem, n_threads
            )
        cp.cuda.get_current_stream().synchronize()
        end_gpu_6.record()
        end_gpu_6.synchronize()
        t6_gpu = (
            (cp.cuda.get_elapsed_time(start_gpu_6, end_gpu_6)) / (n_iterations) * 1e-3
        )

        bench_v1 = benchmark(
            acc.binned_statistic_v1,
            (x_gpu, values_gpu, n_bins, n_threads),
            n_repeat=n_iterations,
        )
        bench_v2 = benchmark(
            acc.binned_statistic_v2,
            (x_gpu, values_gpu, n_bins, n_threads),
            n_repeat=n_iterations,
        )
        bench_v3 = benchmark(
            acc.binned_statistic_v3,
            (x_gpu, values_gpu, n_bins, max_shared_mem, n_threads),
            n_repeat=n_iterations,
        )
        bench_v4 = benchmark(
            acc.binned_statistic_v4,
            (x_gpu, values_gpu, n_bins, max_shared_mem, n_threads),
            n_repeat=n_iterations,
        )
        bench_v5 = benchmark(
            acc.binned_statistic_v5,
            (x_gpu, values_gpu, n_bins, max_shared_mem, n_threads),
            n_repeat=n_iterations,
        )
        bench_v6 = benchmark(
            acc.binned_statistic_v6,
            (x_gpu, values_gpu, n_bins, max_shared_mem, n_threads),
            n_repeat=n_iterations,
        )
        t1_gpu_b = bench_v1.gpu_times.mean()
        t2_gpu_b = bench_v2.gpu_times.mean()
        t3_gpu_b = bench_v3.gpu_times.mean()
        t4_gpu_b = bench_v4.gpu_times.mean()
        t5_gpu_b = bench_v5.gpu_times.mean()
        t6_gpu_b = bench_v6.gpu_times.mean()

    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    start_gpu_1.record()
    for _ in range(n_iterations):
        acc.binned_statistic_v1_dist(
            comm, local_x_gpu, local_values_gpu, n_bins, n_threads
        )
    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    end_gpu_1.record()
    end_gpu_1.synchronize()
    t1_gpus = (cp.cuda.get_elapsed_time(start_gpu_1, end_gpu_1)) / (n_iterations) * 1e-3

    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    start_gpu_2.record()
    for _ in range(n_iterations):
        acc.binned_statistic_v2_dist(
            comm, local_x_gpu, local_values_gpu, n_bins, n_threads
        )
    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    end_gpu_2.record()
    end_gpu_2.synchronize()
    t2_gpus = (cp.cuda.get_elapsed_time(start_gpu_2, end_gpu_2)) / (n_iterations) * 1e-3

    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    start_gpu_3.record()
    for _ in range(n_iterations):
        acc.binned_statistic_v3_dist(
            comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads
        )
    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    end_gpu_3.record()
    end_gpu_3.synchronize()
    t3_gpus = (cp.cuda.get_elapsed_time(start_gpu_3, end_gpu_3)) / (n_iterations) * 1e-3

    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    start_gpu_4.record()
    for _ in range(n_iterations):
        acc.binned_statistic_v4_dist(
            comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads
        )
    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    end_gpu_4.record()
    end_gpu_4.synchronize()
    t4_gpus = (cp.cuda.get_elapsed_time(start_gpu_4, end_gpu_4)) / (n_iterations) * 1e-3

    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    start_gpu_5.record()
    for _ in range(n_iterations):
        acc.binned_statistic_v5_dist(
            comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads
        )
    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    end_gpu_5.record()
    end_gpu_5.synchronize()
    t5_gpus = (cp.cuda.get_elapsed_time(start_gpu_5, end_gpu_5)) / (n_iterations) * 1e-3

    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    start_gpu_6.record()
    for _ in range(n_iterations):
        acc.binned_statistic_v6_dist(
            comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads
        )
    cp.cuda.get_current_stream().synchronize()
    comm.barrier()
    end_gpu_6.record()
    end_gpu_6.synchronize()
    t6_gpus = (cp.cuda.get_elapsed_time(start_gpu_6, end_gpu_6)) / (n_iterations) * 1e-3

    bench_v1_dist = benchmark(
        acc.binned_statistic_v1_dist,
        (comm, local_x_gpu, local_values_gpu, n_bins, n_threads),
        n_repeat=n_iterations,
    )
    bench_v2_dist = benchmark(
        acc.binned_statistic_v2_dist,
        (comm, local_x_gpu, local_values_gpu, n_bins, n_threads),
        n_repeat=n_iterations,
    )
    bench_v3_dist = benchmark(
        acc.binned_statistic_v3_dist,
        (comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads),
        n_repeat=n_iterations,
    )
    bench_v4_dist = benchmark(
        acc.binned_statistic_v4_dist,
        (comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads),
        n_repeat=n_iterations,
    )
    bench_v5_dist = benchmark(
        acc.binned_statistic_v5_dist,
        (comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads),
        n_repeat=n_iterations,
    )
    bench_v6_dist = benchmark(
        acc.binned_statistic_v6_dist,
        (comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads),
        n_repeat=n_iterations,
    )
    # print(bench1_dist, flush=True)
    # print(bench2_dist, flush=True)
    # print(bench3_dist, flush=True)
    # print(bench4_dist, flush=True)
    # print(bench5_dist, flush=True)
    # print(bench6_dist, flush=True)
    t1_gpus_b = bench_v1_dist.gpu_times.mean()
    t2_gpus_b = bench_v2_dist.gpu_times.mean()
    t3_gpus_b = bench_v3_dist.gpu_times.mean()
    t4_gpus_b = bench_v4_dist.gpu_times.mean()
    t5_gpus_b = bench_v5_dist.gpu_times.mean()
    t6_gpus_b = bench_v6_dist.gpu_times.mean()
    if rank == 0:
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime 1 GPU  - v1:{t1_gpu:.6f} s \t[{t1_gpu_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime 1 GPU  - v2:{t2_gpu:.6f} s \t[{t2_gpu_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime 1 GPU  - v3:{t3_gpu:.6f} s \t[{t3_gpu_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime 1 GPU  - v4:{t4_gpu:.6f} s \t[{t4_gpu_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime 1 GPU  - v5:{t5_gpu:.6f} s \t[{t5_gpu_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime 1 GPU  - v6:{t6_gpu:.6f} s \t[{t6_gpu_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime {world_size} GPUs - v1:{t1_gpus:.6f} s \t[{t1_gpus_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime {world_size} GPUs - v2:{t2_gpus:.6f} s \t[{t2_gpus_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime {world_size} GPUs - v3:{t3_gpus:.6f} s \t[{t3_gpus_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime {world_size} GPUs - v4:{t4_gpus:.6f} s \t[{t4_gpus_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime {world_size} GPUs - v5:{t5_gpus:.6f} s \t[{t5_gpus_b:.6f} s]",
            flush=True,
        )
        print(
            f"Size:{n_samples}\tBins:{n_bins}\tTime {world_size} GPUs - v6:{t6_gpus:.6f} s \t[{t6_gpus_b:.6f} s]\n",
            flush=True,
        )
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def main():

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS")))
    # print(f"world_size = {world_size}", flush=True)
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    # print(f"rank = {rank}", flush=True)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID")))
    # print(f"local_rank = {local_rank}", flush=True)
    dev = cp.cuda.Device(local_rank)
    dev.use()
    comm = init_process_group(world_size, rank, use_mpi=True)
    n_samples = 413340001
    if rank == 0:
        x_cpu = (
            0.01 * np.sin(np.linspace(0.0, 5.0, n_samples))
            + np.linspace(0.0, 10.0, n_samples) ** 2
            + 1.0
        )
        values_cpu = (
            np.linspace(0.0, 10.0, n_samples)
            + 0.01 * np.cos(np.linspace(0.0, 5.0, n_samples))
            + 2.0
        )
    else:
        x_cpu = np.zeros(n_samples)
        values_cpu = np.zeros(n_samples)
    n_bins = [433, 833, 1533, 2999, 6999]
    sh_mems = [None, None, None, None, None]
    n_threads = [128, 256, 512, 768, 1024]
    for n_bin, sh_mem, n_thread in zip(n_bins, sh_mems, n_threads):
        bench(comm, x_cpu, values_cpu, n_bin, sh_mem, n_thread)


if __name__ == "__main__":
    main()
