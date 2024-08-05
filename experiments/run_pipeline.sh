#!/bin/bash

#SBATCH -J run_pipeline
#SBATCH -t 00:30:00
#SBATCH --mem=5G
#SBATCH -c 6
#SBATCH -A proj_mscm
#SBATCH -p all.q
#SBATCH -o path/to/output_%A.out

PYTHON="path/to/python"
SCRIPT="path/to/script"
DATA="path/to/data"
RESULT="path/to/result"

if (( $# > 1 ))
then
    SAMPLE_SIZE=$2
else
    SAMPLE_SIZE=1000
fi

cmd="$PYTHON $SCRIPT --outcome $1 --data $DATA --result $RESULT --sample_size $SAMPLE_SIZE"
echo $cmd
eval $cmd
