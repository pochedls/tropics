# -*- coding: utf-8 -*-

"""run_amsu_monthly.py
    
    Wrapper for RTTOV to process a month of hourly AMSU data at a time (in parallel).

    Author: Stephen Po-Chedley
"""

import datetime
from fx import run_amsu_chunk
from joblib import Parallel, delayed
import time
import xarray as xr
import numpy as np
import argparse
import os
from calendar import monthrange

# specify argparse arguments
parser = argparse.ArgumentParser(description='Process arguments for run_amsu_monthly.')
parser.add_argument('year', metavar='y', type=int,
                    help='year of data to process')
parser.add_argument('month', metavar='m', type=int,
                    help='month of data to process')
parser.add_argument('-p', dest='output_path', type=str, required=False,
                    default='/p/vast1/pochedls/era5/amsu/',
                    help='path to save output (default /p/vast1/pochedls/era5/amsu/)')
parser.add_argument('-g', dest='grid', type=str, required=False,
                    default='1x1',
                    help='grid resolution (default "1x1")')
parser.add_argument('-n', dest='nprofs_per_call', type=int, required=False,
                    default=22000,
                    help='number of profiles per call to RTTOV (default 22000)')
parser.add_argument('-t', dest='nthreads', type=int, required=False,
                    default=3,
                    help='number of threads (default 3)')

# get arguments
args = parser.parse_args()
year = args.year
month = args.month
dpath_out = args.output_path
grid = args.grid
nprofs_per_call = args.nprofs_per_call
nthreads = args.nthreads

# get days in month
days = np.arange(1, monthrange(year, month)[1]+1)

print('Start: ' + str(datetime.datetime.now()))
for day in days:
    # print progress
    print('Processing: ' + '-'.join([str(year), str(month), str(day)]))
    # create filename
    fnOut = dpath_out + 'amsu_hourly_' + str(year) + '{:02}'.format(month) + '{:02}'.format(day) + '.nc'
    # skip processing if data exists
    if os.path.exists(fnOut):
        continue
    # get hourly datetime values
    dtvec = [datetime.datetime(int(year), int(month), day, i) for i in range(24)]
    #  parallelize over hourly data
    dsets = Parallel(n_jobs=12)(delayed(run_amsu_chunk)(dt, grid=grid, nprofs_per_call=nprofs_per_call, nthreads=nthreads) for dt in dtvec)
    # concatenate output into daily file
    time = xr.DataArray(name='time', data=dtvec, dims=['time'])
    ds = xr.concat(dsets, dim=time)
    # save output
    ds.to_netcdf(fnOut)
print('Finish: ' + str(datetime.datetime.now()))
