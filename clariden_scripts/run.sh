#!/bin/bash

MODEL=GPT-Neo
script=run_gpt_neo.slurm

################################### 2 nodes
export NODES=2

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 1 1 2 4 1

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 1 1 2 4 1

################################# 4 nodes
export NODES=4

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 64 1 4 4 1

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 64 1 4 4 1

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 64 2 2 4 1

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 64 2 2 4 1

################################# 6 nodes
export NODES=6

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 256 1 6 4 2

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 256 1 6 4 2

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 256 3 2 4 2

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 256 3 2 4 2


################################# 8 nodes

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 1024 2 4 4 4

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 1024 2 4 4 4

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 1024 4 2 4 4

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 1024 4 2 4 4


################################# 16 nodes

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 1024 4 4 4 4

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 1024 4 4 4 4

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 1024 8 2 4 4

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 1024 8 2 4 4


################################# 32 nodes

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 1024 8 4 4 4

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 1024 8 4 4 4

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/time
sbatch -A a-a09 -n $NODES $script 1024 16 2 4 4

export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden/$MODEL/memory
sbatch -A a-a09 -n $NODES $script 1024 16 2 4 4
