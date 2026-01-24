#!/bin/bash -l
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_bench_singularity
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load cuda/12.6
module load openmpi/4.1.6--gcc--12.2.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PMIX_MCA_gds=hash
export PMIX_MCA_psec=native
export OMPI_MCA_btl=^openib

srun -N1 -n4 singularity exec --env  UCX_POSIX_USE_PROC_LINK=n --nv acc_binned_statistic.sif python3 ./benchmarks/bench.py


