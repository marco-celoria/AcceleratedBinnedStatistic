# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import cupy as cp
import numpy as np
import scipy

import acceleratedbinnedstatistic.binned_statistic as acc


def main():
    n_samples = 40000
    n_bins = 15
    arr_cpu = np.linspace(0.0, 4.0, n_samples)

    x_cpu = 0.01 * np.sin(arr_cpu) + arr_cpu**2 + 1.0
    v_cpu = 0.01 * np.cos(arr_cpu) + arr_cpu + 2.0
    statistic_cpu, _, _ = scipy.stats.binned_statistic(x_cpu, v_cpu, bins=n_bins)

    x_gpu = cp.asarray(x_cpu, dtype=cp.float64)
    v_gpu = cp.asarray(v_cpu, dtype=cp.float64)
    statistic_gpu = acc.binned_statistic(x_gpu, v_gpu, n_bins)

    print(f"Results from scipy:\n{statistic_cpu}", flush=True)
    print(f"Results from acceleratedbinnedstatistic\n{statistic_gpu}", flush=True)
    assert np.allclose(statistic_cpu, statistic_gpu.get())


if __name__ == "__main__":
    main()
