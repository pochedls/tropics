# -*- coding: utf-8 -*-

"""create_reference_diurnal_cycle.py
    
    Python Script to create a reference TMT diurnal cycle.

    Author: Stephen Po-Chedley
"""

# %% imports
import xcdat as xc
import xarray as xr
import numpy as np
import datetime
import scipy
from joblib import Parallel, delayed

def get_hourly_inds(time, hour):
    """
    get_hourly_inds(time, hour)

    Function returns the indices for all timesteps within 30 minutes
    of a given hourly time point (irrespective of date, e.g.,
    2020-01-01 00:00, 2020-01-02 00:00, 2020-01-03 00:00 for an hour
    timepoint of hour=0).

    Parameters:
    -----------
    hour (int):            hour of interest (0 - 23)
    time (array-like):     array of cftime objects

    Returns:
    --------
    List : List of indices with time points within 30 minutes of hour
           of interest.
    """
    # get cftime object type
    cfobj = type(time[0])
    # initialize index list
    inds = []
    # loop over all time values in array
    for i, t in enumerate(time):
        # get time components
        y = t.year; m = t.month; d = t.day
        # if hour is midnight, look for timepoints
        # after 23:30 or before 00:30. 
        # else just look for the hour +/- 30 minutes
        # retain indices within +/- 30 minute bounds
        if hour == 0:
            lbnd = cfobj(y, m, d, 23, 30)
            ubnd = cfobj(y, m, d, hour, 30)
            if ((t > lbnd) | (t <= ubnd)):
                inds.append(i)
        else:
            lbnd = cfobj(y, m, d, hour-1, 30)
            ubnd = cfobj(y, m, d, hour, 30)
            if ((t > lbnd) & (t <= ubnd)):
                inds.append(i)
    # return list of indexes
    return inds

def create_hourly_climatology(ds, hour, ngrid):
    """
    create_hourly_climatology(ds, hour, ngrid)

    Function returns the hourly mean values for tmt, tmt_clear, and emissivity
    remapped to a common grid for a given xr.Dataset.

    Parameters:
    -----------
    ds (xr.Dataset):      Dataset containing tmt, tmt_clear, and emissivity
    hour (int):           Hour of interest (0 - 23)
    ngrid (xr.Dataset):   Dataset containing target grid for remapping

    Returns:
    --------
    xr.Dataset: xarray dataset containing hourly mean values for tmt, tmt_clear,
                and emissivity.
    """
    # get hourly indices
    inds = get_hourly_inds(ds.time.values, hour)
    dsh = ds.isel(time=inds)
    # remap data to target grid
    dsr = dsh.regridder.horizontal('tmt', ngrid, method='bilinear', tool='xesmf', periodic=True)
    dsr['tmt_clear'] = dsh.regridder.horizontal('tmt_clear', ngrid, method='bilinear', tool='xesmf', periodic=True)['tmt_clear']
    dsr['emissivity'] = dsh.regridder.horizontal('emissivity', ngrid, method='bilinear', tool='xesmf', periodic=True)['emissivity']
    # take time mean
    dsrm = dsr.mean(dim='time')
    return dsrm


# %% parameters
nlon = np.arange(0.5, 360, 1)  # np.arange(1.25, 360., 2.5)
nlat = np.arange(-89.5, 90., 1)  # np.arange(-88.75, 90, 2.5)
gridlabel = '1x1'  # 2.5x2.5
dpath = '/p/vast1/pochedls/era5/tmt/hourly/'
dpathout = '/p/vast1/pochedls/era5/tmt/climatology/'
years = np.arange(2013, 2023)
ngrid = xc.create_grid(nlat, nlon)

# %% get climatologies for each year
time = []
monthly_climatologies = []
# loop over years and months in each year
for year in years:
    print(year)
    for im, month in enumerate(range(1, 13)):
        print('   ' + str(month))
        # open monthly dataset
        fn = dpath + 'tmt_hourly_' + str(year) + str(month).zfill(2) + '.nc'
        ds = xc.open_dataset(fn)
        ds = ds.drop_vars('longitude_bnds')
        ds = ds.bounds.add_missing_bounds()
        # get mean time coordinate
        time.append(ds.time.mean().values.item() + datetime.timedelta(minutes=30))
        # get regridded climatology for each hour
        hclims = Parallel(backend='multiprocessing', n_jobs=24)(delayed(create_hourly_climatology)(ds, hour, ngrid) for hour in range(0, 24))
        # concatenate 24 hours together for each month
        dsmonth = xr.concat(hclims, dim='hour')
        monthly_climatologies.append(dsmonth)
        ds.close()

# concatenate all monthly climatologies together
dsout = xr.concat(monthly_climatologies, dim='time')
# create time axis for dataset
time = xr.DataArray(data=time, dims=['time'], coords=dict(time=(["time"], time)))
dsout['time'] = time

# save out dataset
if len(years) > 1:
    fnOut = dpathout + 'tmt_reference_monthly_diurnal_' + gridlabel + '_' + str(years[0]) + '-' + str(years[-1]) + '.nc'
else:
    fnOut = dpathout + 'tmt_reference_monthly_diurnal_' + gridlabel + '_' + str(years[0]) + '.nc'
dsout.to_netcdf(fnOut)

# %% now cast dataset to local time

# open dataset
ds = xr.open_dataset(fnOut)

#  create referene time array
reftime = np.array([datetime.datetime(1979, 1, 1, i) for i in range(0, 24)])
hours = np.arange(0, 24)

# get tmt / tmt_clear / emissivity
tmt = ds.tmt.copy(deep=True)
tmt_clear = ds.tmt_clear.copy(deep=True)
emissivity = ds.emissivity.copy(deep=True)

# loop over lines of longitude
for il, lon in enumerate(ds.lon.values):
    print(il)
    # get zonal brightness temperatures
    bt = ds.tmt[:, :, :, il].values
    btc = ds.tmt_clear[:, :, :, il].values
    emis = ds.emissivity[:, :, :, il]
    # get local time offset in seconds
    seconds = lon/15*3600
    # add offset to reference time
    lt = reftime + datetime.timedelta(seconds=seconds)
    # convert to decimal hours
    lt = [t.hour + t.minute/60 + t.second/3600 for t in lt]
    # sort ascending in hours
    inds = np.argsort(lt)
    lt = list(np.array(lt)[inds])
    bt = bt[:, inds, :]
    btc = btc[:, inds, :]
    emis = emis[:, inds, :]
    # pad ends
    lt = [lt[0] - 1] + lt + [lt[-1] + 1]
    lt = np.array(lt)
    bt = np.concatenate((np.expand_dims(bt[:, -1, :], axis=1), bt, np.expand_dims(bt[:, 0, :], axis=1)), axis=1)
    btc = np.concatenate((np.expand_dims(btc[:, -1, :], axis=1), btc, np.expand_dims(btc[:, 0, :], axis=1)), axis=1)
    emis = np.concatenate((np.expand_dims(emis[:, -1, :], axis=1), emis, np.expand_dims(emis[:, 0, :], axis=1)), axis=1)
    # interpolate
    f = scipy.interpolate.interp1d(lt, bt, axis=1)
    fc = scipy.interpolate.interp1d(lt, btc, axis=1)
    fe = scipy.interpolate.interp1d(lt, emis, axis=1)
    bti = f(hours)
    btci = fc(hours)
    emisi = fe(hours)
    # add to array
    tmt[:, :, :, il] = bti
    tmt_clear[:, :, :, il] = btci
    emissivity[:, :, :, il] = emisi

# save output
ds = tmt.to_dataset()
ds['tmt_clear'] = tmt_clear
ds['emissivity'] = emissivity
fnOut = fnOut.replace('_diurnal_', '_diurnal_local_')
ds.to_netcdf(fnOut)

