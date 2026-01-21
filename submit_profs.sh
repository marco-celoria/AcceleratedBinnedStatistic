#!/bin/bash -l
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_prof
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg

export PYTHONPATH="${PWD}/src/"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load cuda/12.1
module load gcc/12.2.0-binut2.41
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.6--gcc--8.5.0
source ./.venv/bin/activate

cd profiles

mkdir -p reports_prof_${SLURM_JOB_ID}

srun -N1 -n4 nsys profile --stats=true -t nvtx,cuda,mpi -o nsys_prof_binstat.%q{SLURM_PROCID}.%q{SLURM_JOBID} python prof.py
#srun -N1 -n4 ncu --replay-mode application   -o ncu_prof_binstat.%q{SLURM_PROCID}.%q{SLURM_JOBID} --kernel-name regex:binned_statistic_kernel python prof.py

mv nsys_prof_binstat*.nsys-rep reports_prof_${SLURM_JOB_ID}
mv nsys_prof_binstat*.sqlite   reports_prof_${SLURM_JOB_ID}

