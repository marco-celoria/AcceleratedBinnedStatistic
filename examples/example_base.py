# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import cupy as cp
import acceleratedbinnedstatistic.binned_statistic as acc
import scipy

def main():
    n_samples = 40000
    x_cpu = 0.01 * np.sin(np.linspace(0., 4., n_samples)) + np.linspace(0., 4., n_samples)**2 + 1.
    x_gpu = cp.asarray(x_cpu, dtype=cp.float64)
    values_cpu = np.linspace(0., 4., n_samples) + 0.01 * np.cos(np.linspace(0., 4., n_samples)) + 2.
    values_gpu = cp.asarray(values_cpu, dtype=cp.float64)
    n_bins = 15
    n_threads = 256
    statistic_gpu = acc.binned_statistic(x_gpu, values_gpu, n_bins) 
    statistic_cpu, _, _ = scipy.stats.binned_statistic(x_cpu, values_cpu, bins=n_bins)
    print(f"Results from acceleratedbinnedstatistic\n{statistic_gpu}", flush=True)
    print(f"Results from scipy:\n{statistic_cpu}", flush=True)
    assert np.allclose(statistic_cpu, statistic_gpu.get()) 

if __name__ == "__main__":
    main()


