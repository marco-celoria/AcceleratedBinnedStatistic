# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import cupy as cp
import scipy.stats as stats
from cupyx.distributed import init_process_group
import os
import gc
import acceleratedbinnedstatistic.binned_statistic as acc
import pytest

def _run_all(comm, x_cpu, values_cpu, n_bins, max_shared_mem = None, n_threads = 256):
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
    
    if rank == 0:
        statistic_v1 = acc.binned_statistic_v1(x_gpu, values_gpu, n_bins, n_threads)
        statistic_v2 = acc.binned_statistic_v2(x_gpu, values_gpu, n_bins, n_threads)
        statistic_v3 = acc.binned_statistic_v3(x_gpu, values_gpu, n_bins, max_shared_mem, n_threads)
        statistic_v4 = acc.binned_statistic_v4(x_gpu, values_gpu, n_bins, max_shared_mem, n_threads)
        statistic_v5 = acc.binned_statistic_v5(x_gpu, values_gpu, n_bins, max_shared_mem, n_threads)
        statistic_v6 = acc.binned_statistic_v6(x_gpu, values_gpu, n_bins, max_shared_mem, n_threads)
        
    statistic_v1_dist = acc.binned_statistic_v1_dist(comm, local_x_gpu, local_values_gpu, n_bins, n_threads)
    statistic_v2_dist = acc.binned_statistic_v2_dist(comm, local_x_gpu, local_values_gpu, n_bins, n_threads)
    statistic_v3_dist = acc.binned_statistic_v3_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    statistic_v4_dist = acc.binned_statistic_v4_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    statistic_v5_dist = acc.binned_statistic_v5_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    statistic_v6_dist = acc.binned_statistic_v6_dist(comm, local_x_gpu, local_values_gpu, n_bins, max_shared_mem, n_threads)
    checks_gpu = cp.zeros(12, dtype=int)
    if rank == 0:
        statistic, _, _ = stats.binned_statistic(x_cpu, values_cpu, statistic = "mean", bins = n_bins)
        statistic[np.isnan(statistic)] = 0.
        checks_gpu[0]  = int(cp.allclose(cp.asarray(statistic), statistic_v1))
        checks_gpu[1]  = int(cp.allclose(cp.asarray(statistic), statistic_v2))
        checks_gpu[2]  = int(cp.allclose(cp.asarray(statistic), statistic_v3))
        checks_gpu[3]  = int(cp.allclose(cp.asarray(statistic), statistic_v4))
        checks_gpu[4]  = int(cp.allclose(cp.asarray(statistic), statistic_v5))
        checks_gpu[5]  = int(cp.allclose(cp.asarray(statistic), statistic_v6))
        checks_gpu[6]  = int(cp.allclose(cp.asarray(statistic), statistic_v1_dist))
        checks_gpu[7]  = int(cp.allclose(cp.asarray(statistic), statistic_v2_dist))
        checks_gpu[8]  = int(cp.allclose(cp.asarray(statistic), statistic_v3_dist))
        checks_gpu[9]  = int(cp.allclose(cp.asarray(statistic), statistic_v4_dist))
        checks_gpu[10] = int(cp.allclose(cp.asarray(statistic), statistic_v5_dist))
        checks_gpu[11] = int(cp.allclose(cp.asarray(statistic), statistic_v6_dist))
       # print(f"DEBUG {checks_gpu}", flush=True)
    comm.broadcast(checks_gpu)
    checks_cpu = checks_gpu.get()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return checks_cpu

def _check_errors(errors, checks, n_elements, n_bins):
    if not checks[0] == 1:
        errors.append(f"Serial v1: Error for {n_elements} elements and {n_bins} bins")
    if not checks[1] == 1:
        errors.append(f"Serial v2: Error for {n_elements} elements and {n_bins} bins")
    if not checks[2] == 1:
        errors.append(f"Serial v3: Error for {n_elements} elements and {n_bins} bins")
    if not checks[3] == 1:
        errors.append(f"Serial v4: Error for {n_elements} elements and {n_bins} bins")
    if not checks[4] == 1:
        errors.append(f"Serial v5: Error for {n_elements} elements and {n_bins} bins")
    if not checks[5] == 1:
        errors.append(f"Serial v6: Error for {n_elements} elements and {n_bins} bins")
    if not checks[6] == 1:
        errors.append(f"Distributed v1: Error for {n_elements} elements and {n_bins} bins")
    if not checks[7] == 1:
        errors.append(f"Distributed v2: Error for {n_elements} elements and {n_bins} bins")
    if not checks[8] == 1:
        errors.append(f"Distributed v3: Error for {n_elements} elements and {n_bins} bins")
    if not checks[9] == 1:
        errors.append(f"Distributed v4: Error for {n_elements} elements and {n_bins} bins")
    if not checks[10] == 1:
        errors.append(f"Distributed v5: Error for {n_elements} elements and {n_bins} bins")
    if not checks[11] == 1:
        errors.append(f"Distributed v6: Error for {n_elements} elements and {n_bins} bins")

class TestBinnedStatistics:
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    dev = cp.cuda.Device(local_rank)
    dev.use()
    comm = init_process_group(world_size, rank, use_mpi=True)

    def test_xxs(self):
        errors = []
        n_samples = 17
        if self.rank == 0:
            x_cpu = np.array([1. ,1. ,2. ,5. ,7., 4.0,6.0,12.0,9.0, 8.8, 3.5 , 9.7, 17.1, 6.6, 4.9, 6.2, 8.3])
            values_cpu = np.array([1.0,1.0,2.0,1.5,3.0,2.5,4.5, 4.0,4.1, 3.3, 12.3, 1.9, 8.7, 12.3, 7.8, 10.1, 5.6])
        else:
            x_cpu = np.zeros(n_samples)
            values_cpu = np.zeros(n_samples)
        n_bins = 3
        checks = _run_all(self.comm,  x_cpu, values_cpu, n_bins)
        _check_errors(errors, checks, n_samples, n_bins)
        # assert no error message has been registered, else print messages
        assert not errors, "errors occured:\n{}".format("\n".join(errors))
    

    def test_xs(self):
        errors = []
        n_samples = 1000
        if self.rank == 0:
            x_cpu = np.random.normal(0., 5., n_samples) + np.linspace(0., 5., n_samples)
            values_cpu = 10 * np.random.uniform(low=0.0, high=1.0, size=n_samples) + np.sin(np.linspace(0., 5., n_samples))
        else:
            x_cpu = np.zeros(n_samples)
            values_cpu = np.zeros(n_samples)
        n_bins = [15, 64, 633]
        for n_bin in n_bins:
            checks = _run_all(self.comm,  x_cpu, values_cpu, n_bin)
            _check_errors(errors, checks, n_samples, n_bin)
        assert not errors, "errors occured:\n{}".format("\n".join(errors))
    

    def test_s(self):
        errors = []
        n_samples = 13333
        if self.rank == 0:
            x_cpu = np.random.normal(0, 5, n_samples) + np.linspace(0., 5., n_samples) + 5 * np.cos(np.linspace(0., 5., n_samples))
            values_cpu = 10 * np.random.uniform(low=0.0, high=1.0, size=n_samples) + np.sin(np.linspace(0., 5., n_samples)) + np.linspace(0., 5., n_samples)**2
        else:
            x_cpu = np.zeros(n_samples)
            values_cpu = np.zeros(n_samples)
        n_bins = [15, 64, 633]
        for n_bin in n_bins:
            checks = _run_all(self.comm,  x_cpu, values_cpu, n_bin)
            _check_errors(errors, checks, n_samples, n_bin)
        assert not errors, "errors occured:\n{}".format("\n".join(errors))
    
    
    def test_m(self):
        errors = []
        n_samples = 33339
        if self.rank == 0:
            x_cpu = np.random.normal(0., 5., n_samples) + np.linspace(0.,1.0, n_samples) + 5 * np.cos(np.linspace(0.,5.,n_samples)) + 2.
            values_cpu = np.random.uniform(low=0., high=10., size=n_samples) + np.sin(np.linspace(0., 5., n_samples)) + np.linspace(0., 5., n_samples)**2 + 3 * np.random.standard_cauchy(n_samples)
        else:
            x_cpu = np.zeros(n_samples)
            values_cpu = np.zeros(n_samples)
        n_bins = [15, 64, 633, 2999]
        for n_bin in n_bins:
            checks = _run_all(self.comm,  x_cpu, values_cpu, n_bin)
            _check_errors(errors, checks, n_samples, n_bin)
        assert not errors, "errors occured:\n{}".format("\n".join(errors))


    def test_l(self):
        errors = []
        n_samples = 163339
        if self.rank == 0:
            x_cpu = np.random.normal(0., 5., n_samples) + np.linspace(0., 1., n_samples) + 5 * np.cos(np.linspace(0., 5., n_samples)) + 2.
            values_cpu = np.random.uniform(low=0., high=10., size=n_samples) + np.sin(np.linspace(0., 5., n_samples)) + np.linspace(0., 5., n_samples)**2 + 3 * np.random.standard_cauchy(n_samples)
        else:
            x_cpu = np.zeros(n_samples)
            values_cpu = np.zeros(n_samples)
        n_bins = [15, 64, 633, 2999, 5999]
        sh_mems = [None, None, None, 96256]
        for n_bin, sh_mem in zip(n_bins, sh_mems):
            checks = _run_all(self.comm,  x_cpu, values_cpu, n_bin, sh_mem)
            _check_errors(errors, checks, n_samples, n_bin)
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    def test_xl(self):
        errors = []
        n_samples = 363340001
        if self.rank == 0:
            x_cpu = np.random.normal(0., 5., n_samples) + np.linspace(0., 1., n_samples) + 5 * np.cos(np.linspace(0., 5., n_samples)) + 2.
            values_cpu = np.random.uniform(low=0., high=10., size=n_samples) + np.sin(np.linspace(0., 5., n_samples)) + np.linspace(0., 5., n_samples)**2 + 3 * np.random.standard_cauchy(n_samples)
        else:
            x_cpu = np.zeros(n_samples)
            values_cpu = np.zeros(n_samples)
        n_bins  = [15, 64, 633, 2999,  5999, 9999]
        sh_mems = [None, None, None, None, None, None]
        #sh_mems = [None, None, None, None, 96256, 163600]
        n_threads = [128, 256, 512, 768, 1024]
        for n_bin, sh_mem, n_thread in zip(n_bins, sh_mems, n_threads):
            checks = _run_all(self.comm,  x_cpu, values_cpu, n_bin, sh_mem, n_thread)
            _check_errors(errors, checks, n_samples, n_bin)
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    def test_xxl(self):
        errors = []
        n_samples = 413340001
        if self.rank == 0:
            x_cpu = 0.01 * np.sin(np.linspace(0., 5., n_samples)) + np.linspace(0., 10., n_samples)**2 + 1.
            values_cpu = np.linspace(0., 10., n_samples) + 0.01 * np.cos(np.linspace(0., 5., n_samples)) + 2.
        else:
            x_cpu = np.zeros(n_samples)
            values_cpu = np.zeros(n_samples)
        n_bins = [15, 64, 633, 2999, 5999]
        sh_mems = [None, None, None, None, None]
        #sh_mems = [None, None, None, 126256, 163600]
        n_threads = [128, 256, 512, 768, 1024]
        for n_bin, sh_mem, n_thread in zip(n_bins, sh_mems, n_threads):
            checks = _run_all(self.comm,  x_cpu, values_cpu, n_bin, sh_mem, n_thread)
            _check_errors(errors, checks, n_samples, n_bin)
        assert not errors, "errors occured:\n{}".format("\n".join(errors))



