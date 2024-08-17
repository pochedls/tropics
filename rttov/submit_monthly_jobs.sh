#!/bin/bash

declare -a monthlist=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")
year=$1

for month in ${monthlist[@]}; do
    sbatch --dependency=singleton --job-name=rt$year slurm_monthly.sh $year $month
done

# https://chuckaknight.wordpress.com/2016/08/02/slurm-scheduling-so-they-run-sequentially/
