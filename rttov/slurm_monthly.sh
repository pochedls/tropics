#!/bin/bash

##### These lines are for Slurm
#SBATCH -p pbatch # Queue name (normal is pbatch, debug is pdebug)
#SBATCH -N 1 # Number of nodes
#SBATCH -n 36 # cores per node
#SBATCH -t 6:00:00 # Wall time HH:MM:SS
#SBATCH --mail-type=ALL
#SBATCH -A cbronze
#SBATCH -o /g/g14/pochedls/tropics/logs/slurm-%j.out # output file
#SBATCH -e /g/g14/pochedls/tropics/logs/slurm-%j.err # output file

## sbatch slurm_monthly.sh 2022 1

year=$1
month=$2

. /usr/workspace/pochedls/bin/miniconda3/etc/profile.d/conda.sh

conda activate rttov

cd /g/g14/pochedls/tropics/rttov/

python run_amsu_monthly.py $year $month
