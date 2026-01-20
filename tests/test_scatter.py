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


class TestBinnedStatistics:
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    dev = cp.cuda.Device(local_rank)
    dev.use()
    comm = init_process_group(world_size, rank, use_mpi=True)

    
    def test_scatter_xxs(self):
        n_samples = 17
        if self.rank == 0:
            x_gpu = cp.array([1. ,1. ,2. ,5. ,7., 4.0,6.0,12.0,9.0, 8.8, 3.5 , 9.7, 17.1, 6.6, 4.9, 6.2, 8.3])
        else:
            x_gpu = cp.zeros(n_samples)
        local_x_gpu_0 = acc.scatter_v0(self.comm, x_gpu)
        local_x_gpu_1 = acc.scatter_v1(self.comm, x_gpu)
        assert cp.allclose(local_x_gpu_0, local_x_gpu_1)

    
    def test_scatter_xs(self):
        n_samples = 1000
        if self.rank == 0:
            x_gpu = cp.random.normal(0., 5., n_samples) + cp.linspace(0., 5., n_samples)
        else:
            x_gpu = cp.zeros(n_samples)
        local_x_gpu_0 = acc.scatter_v0(self.comm, x_gpu)
        local_x_gpu_1 = acc.scatter_v1(self.comm, x_gpu)
        assert cp.allclose(local_x_gpu_0, local_x_gpu_1)

    
    def test_scatter_s(self):
        n_samples = 13333
        if self.rank == 0:
            x_gpu = cp.random.normal(0, 5, n_samples) + cp.linspace(0., 5., n_samples) + 5 * cp.cos(cp.linspace(0., 5., n_samples))
        else:
            x_gpu = cp.zeros(n_samples)
        local_x_gpu_0 = acc.scatter_v0(self.comm, x_gpu)
        local_x_gpu_1 = acc.scatter_v1(self.comm, x_gpu)
        assert cp.allclose(local_x_gpu_0, local_x_gpu_1)
    

    def test_scatter_m(self):
        n_samples = 33339
        if self.rank == 0:
            x_gpu = cp.random.normal(0., 5., n_samples) + cp.linspace(0.,1.0, n_samples) + 5 * cp.cos(cp.linspace(0.,5.,n_samples)) + 2.
        else:
            x_gpu = cp.zeros(n_samples)
        local_x_gpu_0 = acc.scatter_v0(self.comm, x_gpu)
        local_x_gpu_1 = acc.scatter_v1(self.comm, x_gpu)
        assert cp.allclose(local_x_gpu_0, local_x_gpu_1)



    def test_scatter_l(self):
        n_samples = 163339
        if self.rank == 0:
            x_gpu = cp.random.normal(0., 5., n_samples) + cp.linspace(0.,1.0, n_samples) + 5 * cp.cos(cp.linspace(0.,5.,n_samples)) + 2.
        else:
            x_gpu = cp.zeros(n_samples)
        local_x_gpu_0 = acc.scatter_v0(self.comm, x_gpu)
        local_x_gpu_1 = acc.scatter_v1(self.comm, x_gpu)
        assert cp.allclose(local_x_gpu_0, local_x_gpu_1)


    def test_scatter_xl(self):
        n_samples = 363340001
        if self.rank == 0:
            x_gpu = cp.random.normal(0., 5., n_samples) + cp.linspace(0.,1.0, n_samples) + 5 * cp.cos(cp.linspace(0.,5.,n_samples)) + 2.
        else:
            x_gpu = cp.zeros(n_samples)
        local_x_gpu_0 = acc.scatter_v0(self.comm, x_gpu)
        local_x_gpu_1 = acc.scatter_v1(self.comm, x_gpu)
        assert cp.allclose(local_x_gpu_0, local_x_gpu_1)


    def test_scatter_xxl(self):
        n_samples = 413340001
        if self.rank == 0:
            x_gpu = 0.01 * cp.sin(cp.linspace(0., 5., n_samples)) + cp.linspace(0., 10., n_samples)**2 + 1.
        else:
            x_gpu = cp.zeros(n_samples)
        local_x_gpu_0 = acc.scatter_v0(self.comm, x_gpu)
        local_x_gpu_1 = acc.scatter_v1(self.comm, x_gpu)
        assert cp.allclose(local_x_gpu_0, local_x_gpu_1)


