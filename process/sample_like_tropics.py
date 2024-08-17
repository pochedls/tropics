# %% imports
import xarray as xr
import numpy as np
import datetime
import scipy
import os
import cftime
from calendar import monthrange
from joblib import Parallel, delayed

def get_tropics_sampling_data(sat, year, doy, sat_day_shift, sat_lon_shift, dpath_tropics='/p/vast1/pochedls/tropics/tropics_x22_raan180_v2/'):
    """
    get_tropics_sampling_data(sat, year, doy, sat_day_shift, sat_lon_shift)

    Function retrieves the tropics sampling points for a given satellite and time. It also
    enables the creation of additional data by shifting the sampling day or longitude value.

    Parameters:
    -----------
    sat (int)                       : satellite id (0 - 3) 
    year (int)                      : year of sampling (to create synthetic cftime object)
    doy (int)                       : day of year
    sat_day_shift (int)             : shift to apply to the sampling day
    sat_lon_shift (float)           : shift to apply to the sampled longitude point
    dpath_tropics (str, optional)   : path to the TROPICS sampling data


    Returns:
    --------
    time (List)                     : cftime objects for each sample
    time_sat_seconds (np.array)     : time (in seconds after midnight) for each sample
    scanlat (np.array)              : array of scan latitude values [time, footprint]
    scanlon (np.array)              : array of scan longitude values [time, footprint]
    
    Note:
    -----
    The TROPICS sampling data is available for four satellites (00, 01, 02, 03) and for
    days 1 - 365. The timestamps here are synthetic based on the user-specified year and
    day of year.

    """
    # get datetime object
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy-1)
    # get doy (including shift)
    doy = doy + sat_day_shift
    if doy > 365:
        doy = doy - 365
    # get satellite file
    fnsat = dpath_tropics + str(sat).zfill(2) + '/tropics_x22_raan180_v2_' + str(sat).zfill(2) + '_DAY_' + str(doy).zfill(3) + '.nc'
    # open tmt file
    dss = xr.open_dataset(fnsat)
    # get tropics lat/lon/time points
    scanlat = dss.LAT.values
    scanlon = dss.LON.values
    scanlon[scanlon < 0] = scanlon[scanlon < 0] + 360.
    # add sat_lon_shift
    scanlon = scanlon + sat_lon_shift
    scanlon[scanlon > 360.] = scanlon[scanlon > 360] - 360.
    time_sat_seconds = dss.TIME.values
    # get datetime objects for time axis
    time = []
    for t in time_sat_seconds:
        dti = dt + datetime.timedelta(milliseconds=t*1000)
        dtic = cftime.datetime(dti.year, dti.month, dti.day, hour=dti.hour, minute=dti.minute, second=dti.second, microsecond=dti.microsecond, calendar='standard')
        time.append(dtic)
    return time, time_sat_seconds, scanlat, scanlon

def get_tmt_data(year, doy, dpath_tmt='/p/vast1/pochedls/era5/tmt/hourly/'):
    """
    get_tmt_data(year, doy)

    Function gets the hourly tmt field for a given day of year (padded with midnight
    of the following day).

    Parameters:
    -----------
    year (int)                  : year of interest
    doy (int)                   : day of year of interest
    dpath_tmt (str, optional)   : directory of tmt data (default: /p/vast1/pochedls/era5/tmt/hourly/)


    Returns:
    --------
    xr.DataArray    : TMT field
    xr.DataArray    : Clear-sky TMT field
    
    """
    # get datetime object
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy-1)
    if dt.day == monthrange(dt.year, dt.month)[1]:
        # on border of month, need next month's data, too
        dtn = dt + datetime.timedelta(days=1)
        fnera = dpath_tmt + 'tmt_hourly_' + str(dt.year) + str(dt.month).zfill(2) + '.nc'
        fnera2 = dpath_tmt + 'tmt_hourly_' + str(dtn.year) + str(dtn.month).zfill(2) + '.nc'
        # if the next months data doesn't exist continue with data in hand
        if not os.path.exists(fnera2):
            dse = xr.open_dataset(fnera, use_cftime=True)
        else:
            fnera = [fnera, fnera2]
            dse = xr.open_mfdataset(fnera, use_cftime=True)
    else:
        fnera = dpath_tmt + 'tmt_hourly_' + str(dt.year) + str(dt.month).zfill(2) + '.nc'
        dse = xr.open_dataset(fnera, use_cftime=True)
    # subset to daily data
    dse = dse.sel(time=slice(dt, dt+datetime.timedelta(days=1))).load()
    tmt = dse.tmt
    tmt_clear = dse.tmt_clear
    return tmt, tmt_clear



# get sat doy (with shift)
def sample_like_tropics(tmt, tmt_clear, time_sat_seconds, time, scanlat, scanlon):
    """
    sample_like_tropics(tmt, tmt_clear, time_sat_seconds, time, scanlat, scanlon)

    Function samples the clear and full-sky TMT fields using TROPICS sampling data.

    Parameters:
    -----------
    tmt (xr.DataArray)              : TMT field
    tmt_clear (xr.DataArray)        : Clear-sky TMT field
    time_sat_seconds (array-like)   : TROPICS sample time (in seconds after 
                                      midnight) for each sample
    time (np.array)                 : TROPICS sample time
    scanlat (np.array)              : TROPICS sampling latitude values [time, footprint]
    scanlon (np.array)              : TROPICS sampling longitude values [time, footprint]

    Returns:
    --------
    time (array-like)               : TROPICS time points
    scanlat (np.array)              : TROPICS sampled latitude points [time, footprint]
    scanlon (np.array)              : TROPICS sampled longitude points [time, footprint]
    tmt_sampled (np.array)          : TMT sampled at TROPICS points [time, footprint]
    tmt_clear_sampled (np.array)    : Clear-sky TMT sampled at TROPICS points [time, footprint]

    Notes:
    ------
    The time, scanlat, and scanlon may change size if TMT data is not available
    for the following day (preventing interpolation to all satellite sample
    points).
    
    """    
    # get era5 time in seconds
    tref = tmt.time.values[0]
    trefdt = datetime.datetime(tref.year, tref.month, tref.day)
    time_era_seconds = [(t - trefdt).total_seconds() for t in tmt.time.values]
    # check if we have the 00:00 of the next day for interpolation
    # if not, drop the last hour of tropics sampling data
    if tmt.time.values[0] + datetime.timedelta(days=1) != tmt.time.values[-1]:
        inds = np.where(time_sat_seconds < 86400 - 3600)[0]
        scanlat = scanlat[inds, :]
        scanlon = scanlon[inds, :]
        time_sat_seconds = time_sat_seconds[inds]
        time = [time[i] for i in inds]
    # create periodic boundary conditions
    tmtLower = tmt[:, :, 0]
    tmtLower['longitude'] = 360.
    tmt = xr.concat((tmt, tmtLower), 'longitude')
    tmtClearLower = tmt_clear[:, :, 0]
    tmtClearLower['longitude'] = 360.
    tmt_clear = xr.concat((tmt_clear, tmtClearLower), 'longitude')
    # pre-allocate output matrices
    tmt_sampled = np.zeros(scanlat.shape)
    tmt_clear_sampled = np.zeros(scanlat.shape)
    # loop over satellite footprints and interpolate
    for fp in range(scanlat.shape[1]):
        # get lat/lon points for footprint
        lat = scanlat[:, fp]
        lon = scanlon[:, fp]
        # construct array of time / lat / lon points [satellite locations]
        slocs = np.zeros((scanlat.shape[0], 3))
        slocs[:, 0] = time_sat_seconds
        slocs[:, 1] = lat
        slocs[:, 2] = lon
        # interpolate clear/full sky tmt to satellite sample locations
        tmtd = scipy.interpolate.interpn((time_era_seconds, tmt.latitude.values, tmt.longitude.values), tmt.values, slocs)
        tmtdc = scipy.interpolate.interpn((time_era_seconds, tmt_clear.latitude.values, tmt_clear.longitude.values), tmt_clear.values, slocs)
        # place interpolated points into sampled arrays
        tmt_sampled[:, fp] = tmtd
        tmt_clear_sampled[:, fp] = tmtdc
    return time, scanlat, scanlon, tmt_sampled, tmt_clear_sampled

def write_sample_output(fnOut, time, scanlat, scanlon, tmt, tmt_clear):
    """
    write_sample_output(fnOut, time, scanlat, scanlon, tmt, tmt_clear)

    Function writes output for TROPICS-sampled TMT data.

    Parameters:
    -----------
    fnOut (str)             : target filename
    time (array-like)       : sampled time points
    scanlat (np.array)      : sampled scan latitude points [time, footprint]
    scanlon (np.array)      : sampled scan longitude points [time, footprint]
    tmt (np.array)          : sampled TMT data [time, footprint]
    tmt_clear (np.array)    : sampled clear-sky TMT data [time, footprint]
    """
    # create lat/lon dataarray
    scanlon = xr.DataArray(
        data=scanlon,
        dims=["time", "footprint"],
        name='scanlon',
        coords=dict(
            footprint=(['footprint'], np.arange(scanlon.shape[1])),
            time=time,
        ),
        attrs=dict(
            description="scan longitude",
            units="degrees_east",
        ),
    )
    scanlat = xr.DataArray(
        data=scanlat,
        dims=["time", "footprint"],
        name='scanlat',
        coords=dict(
            footprint=(['footprint'], np.arange(scanlon.shape[1])),
            time=time,
        ),
        attrs=dict(
            description="scan latitude",
            units="degrees_north",
        ),
    )
    # tmt data to dataarray
    da = xr.DataArray(
        data=tmt,
        dims=["time", "footprint"],
        name='tmt',
        coords=dict(
            scanlon=scanlon,
            scanlat=scanlat,
            time=time,
        ),
        attrs=dict(
            description="brightness_temperature",
            units="K",
        ),

    )
    dac = xr.DataArray(
        data=tmt_clear,
        dims=["time", "footprint"],
        name='tmt_clear',
        coords=dict(
            scanlon=scanlon,
            scanlat=scanlat,
            time=time,
        ),
        attrs=dict(
            description="brightness_temperature_clear",
            units="K",
        ),

    )
    # set time units
    da.time.encoding['units'] = 'hours since 1979-01-01 00:00'
    dac.time.encoding['units'] = 'hours since 1979-01-01 00:00'
    ds = da.to_dataset()
    ds['tmt_clear'] = dac
    ds.to_netcdf(fnOut)


def sample_day(sat, year, doy, sat_day_shift, sat_lon_shift, sat_label, dpathout='/p/vast1/pochedls/era5/tropics/'):
    """
    sample_day(sat, year, doy, sat_day_shift, sat_lon_shift, sat_label, dpathout)

    For a given day of year (1-365) and year, this function will retrieve
    the TROPICS sampling data and the TMT data, sample the TMT data (using
    the TROPICS sampling points), and save the result to a file.

    Parameters:
    -----------
    sat (int)                   : TROPICS satellite to use for sampling data
    year (int)                  : year of interest
    doy (int)                   : day of year of interest
    sat_day_shift (int)         : number of days to shift TROPICS sampling data
    sat_lon_shift (float)       : degrees to shift TROPICS longitude data
    sat_label (str)             : satellite label for output filename
    dpathout (str, optional)    : directory to save data (default /p/vast1/pochedls/era5/tropics/)

    Notes:
    ------
    Function calls get_tropics_sampling_data, get_tmt_data, sample_like_tropics,
    and write_sample_output. The underlying sampling data can be altered using
    sat_day_shift and sat_lon_shift. The output filename follows the form:
    DIRECTORY/tmt_daily_sat_SATNUM_YEARMMDD.nc
    """
    # get datetime object
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy-1)
    # check if file exists
    fnOut = dpathout + 'tmt_daily_sat' + str(sat_label).zfill(2) + '_' + str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + '.nc'
    time, time_sat_seconds, scanlat, scanlon = get_tropics_sampling_data(sat, year, doy, sat_day_shift, sat_lon_shift)
    tmt, tmt_clear = get_tmt_data(year, doy)
    time, scanlat, scanlon, tmt_sampled, tmt_clear_sampled = sample_like_tropics(tmt, tmt_clear, time_sat_seconds, time, scanlat, scanlon)
    write_sample_output(fnOut, time, scanlat, scanlon, tmt_sampled, tmt_clear_sampled)


# parameters
sat_shifts = {2018: 30, 2019: 60, 2020: 90, 2021: 120, 2022: 150}  # shifts to apply each year
dpathout='/p/vast1/pochedls/era5/tropics/'  # output directory
years = np.arange(2018, 2023)  # years to sample
# satellite dictionary with the following form:
#           satlabel: {sat_no, sat_lon_shift}
sats = {0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (3, 0),
        4: (0, 180),
        5: (2, 180)}

# get all parameters
parameter_list = []
# loop over TROPICS satellites
for sat_label in sats.keys():
    # get sat number to load and sat shift to apply
    sat_no, sat_lon_shift = sats[sat_label]
    # loop over years
    for year in years:
        # get the satellite day shift to apply
        sat_day_shift = sat_shifts[year]
        # loop over all days
        for doy in range(1, 366):
            # get datetime object
            dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy-1)
            # check if file exists
            fnOut = dpathout + 'tmt_daily_sat' + str(sat_label).zfill(2) + '_' + str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) + '.nc'
            # skip computation if file exists
            if os.path.exists(fnOut):
                continue
            # get values for sample_day
            values = [sat_no, year, doy, sat_day_shift, sat_lon_shift, sat_label]
            parameter_list.append(values)

# use values from above loop to perform sampling calculation in parallel
print('Start sampling (in parallel)')
print(datetime.datetime.now())
_ = Parallel(n_jobs=20)(delayed(sample_day)(*values) for values in parameter_list)
print(datetime.datetime.now())
