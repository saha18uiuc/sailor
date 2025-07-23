# k1: RTX-3090; n2-n3: Titan-RTX; r4-r6: RTX-2080
################################### 2 nodes

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 1 2 2 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 1 2 2 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 1 2 2 2
###### n1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 1 2 2 2

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 2 1 2 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 2 1 2 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 2 1 2 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 2 1 2 2

################################### 3 nodes

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 144 1 3 8 2
###### r4 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 144 1 3 8 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 144 1 3 8 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 144 1 3 8 2
###### r4
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 144 1 3 8 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 144 1 3 8 2

################################### 4 nodes

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 1 4 8 2
###### r4 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 1 4 8 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 256 1 4 8 2
###### r5 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 256 1 4 8 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 1 4 8 2
###### r4
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 1 4 8 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 256 1 4 8 2
###### r5
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 256 1 4 8 2

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 2 2 8 2
###### r4 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 2 2 8 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 256 2 2 8 2
###### r5 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 256 2 2 8 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 2 2 8 2
###### r4
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 2 2 8 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 256 2 2 8 2
###### r5
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 256 2 2 8 2

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 4 1 8 2
###### r4 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 4 1 8 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 256 4 1 8 2
###### r5 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 256 4 1 8 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 256 4 1 8 2
###### r4
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 256 4 1 8 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 256 4 1 8 2
###### r5
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 256 4 1 8 2

################################### 6 nodes

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 288 2 3 8 2
###### r4 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 288 2 3 8 2
###### r5 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 288 2 3 8 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 288 2 3 8 2
###### r6 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 4 288 2 3 8 2
###### n3 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 5 288 2 3 8 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 288 2 3 8 2
###### r4
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 288 2 3 8 2
###### r5
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 288 2 3 8 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 288 2 3 8 2
###### r6
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 4 288 2 3 8 2
###### n3
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 5 288 2 3 8 2

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 288 3 2 8 2
###### r4 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 288 3 2 8 2
###### r5 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 288 3 2 8 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 288 3 2 8 2
###### r6 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 4 288 3 2 8 2
###### n3 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 5 288 3 2 8 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 288 3 2 8 2
###### r4
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 288 3 2 8 2
###### r5
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 288 3 2 8 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 288 3 2 8 2
###### r6
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 4 288 3 2 8 2
###### n3
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 5 288 3 2 8 2

###### k1 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 288 6 1 8 2
###### r4 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 288 6 1 8 2
###### r5 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 288 6 1 8 2
###### n2 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 288 6 1 8 2
###### r6 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 4 288 6 1 8 2
###### n3 time measurements
export PRINT_MEMORY=0
bash scripts/aceso_opt_execute.sh 18.25.7.1 5 288 6 1 8 2

###### k1
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 0 288 6 1 8 2
###### r4
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 1 288 6 1 8 2
###### r5
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 2 288 6 1 8 2
###### n2
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 3 288 6 1 8 2
###### r6
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 4 288 6 1 8 2
###### n3
export PRINT_MEMORY=1
bash scripts/aceso_opt_execute.sh 18.25.7.1 5 288 6 1 8 2