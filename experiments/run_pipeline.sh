#!/bin/bash

#SBATCH -J run_pipeline
#SBATCH -t 00:30:00
#SBATCH --mem=5G
#SBATCH -c 6
#SBATCH -A proj_mscm
#SBATCH -p all.q
#SBATCH -o path/to/output_%A.out

PYTHON="/ihme/code/mscm/miniconda3/envs/mrtool_0.0.1/bin/python"
SCRIPT="/mnt/team/msca/pub/help-desk/MSCA-308/code/Temperature/experiments/run_pipeline.py"
# this is a copy from
# /mnt/share/erf/temperature/MSCA/mrBrt_R_meanTempDegree_adm1_dailyTemp_refzoneMean_2-28_allLocations.csv
DATA="/mnt/team/msca/pub/help-desk/MSCA-308/data/data.parquet"
RESULT="/mnt/team/msca/pub/help-desk/MSCA-308/result_ckd"

if (( $# > 1 ))
then
    SAMPLE_SIZE=$2
else
    SAMPLE_SIZE=1000
fi

cmd="$PYTHON $SCRIPT --outcome $1 --data $DATA --result $RESULT --n_samples $SAMPLE_SIZE"
echo $cmd
eval $cmd
