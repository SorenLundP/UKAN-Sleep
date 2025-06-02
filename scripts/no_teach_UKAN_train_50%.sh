#!/bin/bash -l

#SBATCH --partition=small-g
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=400G
#SBATCH --cpus-per-task=32
#SBATCH --account=project_4650XXXXX


module load CrayEnv
module load cotainr

# Setup temporary directories for cotainr/Singularity
mkdir -p /tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

srun singularity exec --mount type=bind,src=/scratch/project_4650XXXXX,dst=/users/XXXX/mnt \
  /users/XXXX/cotainrImage.sif python /users/XXXX/USleep_KAN_Gram/No_teach_gram_train_50%.py
