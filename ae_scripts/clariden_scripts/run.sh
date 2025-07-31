!/bin/bash

MODEL=OPT-350
script=clariden_scripts/run_opt.slurm

################################### 1 node
export NODES=1

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1 1 1 4 1

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1 1 1 4 1

######################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 32 1 1 4 1

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 32 1 1 4 1


################################### 2 nodes
export NODES=2

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 32 1 2 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 32 1 2 4 8

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 32 2 1 4 2

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 32 2 1 4 2

################################### 4 nodes
export NODES=4

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 64 1 4 4 1

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 64 1 4 4 1

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 64 2 2 4 1

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 64 2 2 4 1

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 128 4 1 4 4

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 128 4 1 4 4

################################### 8 nodes
export NODES=8

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 512 2 4 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 512 2 4 4 8

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 512 4 2 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 512 4 2 4 8

################################### 16 nodes
export NODES=16

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 16 1 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 16 1 4 8

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 4 4 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 4 4 4 8

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 8 2 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 8 2 4 8

################################### 32 nodes
export NODES=32

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 16 2 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 16 2 4 8

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 32 1 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 32 1 4 8

##################

# time measurements
export PRINT_MEMORY=0
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/time
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 8 4 4 8

# memory
export PRINT_MEMORY=1
export BASE_DIR=/capstor/scratch/cscs/$USER/sailor/clariden/$MODEL/memory
sbatch -t 5:00:00 -A a-infra02 -n $NODES $script 1024 8 4 4 8
