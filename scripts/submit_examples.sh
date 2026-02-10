#!/bin/bash -l
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --mem=0
#SBATCH --job-name=run_example
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg

cd ..

export PYTHONPATH="src/"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load cuda/12.6
module load gcc/12.2.0-binut2.41
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.6--gcc--8.5.0
source ./.venv/bin/activate


srun -N1 -n1 python examples/example_base.py
srun -N1 -n4 python examples/example_dist.py

