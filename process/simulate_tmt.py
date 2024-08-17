# %% imports
import xarray as xr
import numpy as np
import glob
import os
from calendar import monthrange
from joblib import Parallel, delayed

def amsu_to_tmt(year, month, dpathout='/p/vast1/pochedls/era5/tmt/hourly/', dpath='/p/vast1/pochedls/era5/amsu/'):
    """
    amsu_to_tmt(year, month)

    Function reads in simulated AMSU data and writes out TMT data.

    Parameters:
    -----------
    year (int)                  : Year of data to process
    month (int)                 : Month of data to process
    dpathout (str, optional)    : Output directory to save data (default /p/vast1/pochedls/era5/tmt/hourly/)
    dpath (str, optional)       : Source directory (default /p/vast1/pochedls/era5/amsu/)

    Notes:
    ------
    Function assumes input files follow the form (amsu_hourly_YYYYMMDD.nc) with
    variables bt and bt_clear of the form [time, channel, view_angle, lat, lon].
    The function assumes TMT is the average of the 12 central view angles. Output
    files are saved as tmt_hourly_YYYYMM.nc. If output file exists, no computation
    is performed.
    """
    # get number of days in month
    dim = monthrange(year, month)[1]
    # ensure we have expected number of files
    filestr = dpath + 'amsu_hourly_' + str(year) + str(month).zfill(2) + '*.nc'
    files = glob.glob(filestr)
    if len(files) != dim:
        raise ValueError('Expected ' + str(dim) + ' files for ' + str(year) + '/' + str(month) + '. Found ' + str(len(files)) + '.')
    # check if output file exists
    fnOut = dpathout + 'tmt_hourly_' + str(year) + str(month).zfill(2) + '.nc'
    if os.path.exists(fnOut):
        return
    # loop over files and convert to tmt
    tmt = []
    tmt_clear = []
    emissivity = []
    for fn in files:
        # open file
        ds = xr.open_dataset(fn, use_cftime=True)
        # average over channel 5
        ds5 = ds.sel(channel=5)
        ds5 = ds5.isel(view_angle=[0, 1, 2, 3, 4, 5])
        ds5 = ds5.mean(dim='view_angle', keep_attrs=True)
        # set to tmt
        ds5['tmt'] = ds5['bt'].rename('tmt')
        ds5['tmt_clear'] = ds5['bt_clear'].rename('tmt_clear')
        ds5 = ds5.drop_vars('bt')
        ds5 = ds5.drop_vars('bt_clear')
        # save out data
        time = ds5.time
        time.encoding['units'] = 'hours since 1979-01-01 00:00:00'
        ds5['time'] = time
        tmt.append(ds5.tmt)
        tmt_clear.append(ds5.tmt_clear)
        emissivity.append(ds5.emissivity)
        del ds5
    # concatenate file data
    tmt = xr.concat(tmt, dim='time')
    tmt_clear = xr.concat(tmt_clear, dim='time')
    emissivity = xr.concat(emissivity, dim='time')
    # save results as netcdf dataset
    dsout = tmt.to_dataset()
    dsout['tmt_clear'] = tmt_clear
    dsout['emissivity'] = emissivity
    dsout.to_netcdf(fnOut)

# %% Parameters
dpath = '/p/vast1/pochedls/era5/amsu/'
dpathout = '/p/vast1/pochedls/era5/tmt/hourly/'
years = np.arange(2007, 2013)
months = np.arange(1, 13)

# loop over years and convert (all months in parallel) amsu data to tmt
for year in years:
    print(year)
    Parallel(n_jobs=12)(delayed(amsu_to_tmt)(year, month, dpathout=dpathout) for month in months)

