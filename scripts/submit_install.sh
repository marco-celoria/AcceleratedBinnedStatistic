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

cd ..

BASE_DIR=$(pwd)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load cuda/12.6
module load gcc/12.2.0-binut2.41  
module load openmpi/4.1.6--gcc--12.2.0 
module load python/3.11.6--gcc--8.5.0

python -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -U setuptools pip
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
python -m cupyx.tools.install_library --cuda 12.x --library cutensor --prefix "${BASE_DIR}/.venv/.cupy/cuda_lib"
python -m cupyx.tools.install_library --cuda 12.x --library nccl     --prefix "${BASE_DIR}/.venv/.cupy/cuda_lib"
python -m cupyx.tools.install_library --cuda 12.x --library cudnn    --prefix "${BASE_DIR}/.venv/.cupy/cuda_lib"
