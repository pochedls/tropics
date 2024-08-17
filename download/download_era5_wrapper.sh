#!/bin/bash

# running on oslic3

# include certificate on path
export REQUESTS_CA_BUNDLE="./cspca.crt"

# activate correct conda shell
eval "$(conda shell.bash hook)"
conda deactivate
conda activate cds


echo "$(date)"

for month in {1..12}
do
    python download_era5.py $1 $month
done


echo "$(date)"