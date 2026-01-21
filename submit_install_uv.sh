#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=lrd_all_serial
#SBATCH --time=4:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=job_install
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export UV_LINK_MODE=copy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load uv
module load cuda/12.1
module load gcc/12.2.0-binut2.41  
module load openmpi/4.1.6--gcc--12.2.0 
module load python/3.11.6--gcc--8.5.0

uv venv --no-managed-python .venv
source .venv/bin/activate
uv sync
python -m cupyx.tools.install_library --cuda 12.x --library cutensor --prefix "${SLURM_SUBMIT_DIR}/.venv/.cupy/cuda_lib"
python -m cupyx.tools.install_library --cuda 12.x --library nccl     --prefix "${SLURM_SUBMIT_DIR}/.venv/.cupy/cuda_lib"
python -m cupyx.tools.install_library --cuda 12.x --library cudnn    --prefix "${SLURM_SUBMIT_DIR}/.venv/.cupy/cuda_lib"

make install
