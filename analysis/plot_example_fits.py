# -*- coding: utf-8 -*-

"""plot_satellites.py
    
    Python script to plot satellite paths and sampling.

    Author: Stephen Po-Chedley
"""

# %% imports
import xarray as xr
import numpy as np
import glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import datetime

# %% helper functions
def get_sat_data(sat, month, years, fprints=[19, 29, 39, 49, 59]):
    """
    get_sat_data(sat, day, span, pday)


    Parameters:
    -----------


    Returns:
    --------

    """
    dpath = '/p/vast1/pochedls/era5/tropics/'
    files = []
    for year in years:
        yfiles = glob.glob(dpath + 'tmt_daily_sat' + str(sat).zfill(2) + '_' + str(year) + str(month).zfill(2) + '*.nc')
        files = files + yfiles
    ds = xr.open_mfdataset(files, use_cftime=True)
    ds = ds.isel(footprint=fprints).load()
    tmt = np.reshape(np.array(ds.tmt.values), -1)
    tmt_clear = np.reshape(np.array(ds.tmt_clear.values), -1)
    lon = np.reshape(np.array(ds.scanlon.values), -1)
    lon[lon < 0] = lon[lon < 0] + 360.
    lat = np.reshape(np.array(ds.scanlat.values), -1)
    time = np.tile(np.expand_dims(ds.time.values, axis=1), (1, len(fprints)))
    time = np.reshape(np.array(time), -1)
    ltime = np.array([get_local_time(t, l) for t, l in zip(time, lon)])
    ds.close()
    return ltime, lat, lon, tmt, tmt_clear


def get_local_time(utctime, lon):
    """
    get_local_time(utctime, lon)

    Function takes a datetime object (in UTC time) and converts it
    to the hour of the day (i.e., 0 - 23.999) using the sampling
    point of the longitude value.

    Parameters:
    -----------
    utctime (datetime object)   : datetime object (in UTC)
    lon (float)                 : longitude value of point of interest

    Returns:
    --------
    float : hour of the day
    """
    tl = utctime.hour + utctime.minute/60. + utctime.second/3600
    tl = np.mod(tl + lon / 15., 24.)
    return tl


def fourier(time, a0, a1, a2, t1, t2):
    """
    fourier(time, a0, a1, a2, t1, t2)

    Function is a second-order Fourier series to be used with
    the scipy curve_fit utility.

    Parameters:
    -----------
    time (array-like) : time in units of hour of day (float values)
    a0 (float)        : a0 parameter (see below)
    a1 (float)        : a1 parameter (see below)
    a2 (float)        : a2 parameter (see below)
    t1 (float)        : t1 parameter (see below)
    t2 (float)        : t2 parameter (see below)

    Returns:
    --------
    Result from second-order Fourier series with user-specified values.

    Notes:
    ------
    Example use in Lindfors et al., 2011 (doi: 10.1175/JTECH-D-11-00093.1).
    Functional form:
        a0 + a1 * cos(pi * (time - t1) / 12) + a2 * cos(2*pi * (time - t2 / 12))
    """
    return a0 + a1 * np.cos(np.pi * (time - t1)/12.) + a2 * np.cos(2 * np.pi * (time - t2)/12.)


# %% parameters
plocs = [[-8.5, 242.5], [-11.5, 308.5], [-23.5, 344.5], [24.5, 17.5], [29.5, 90.5], [-29.5, 176.5]]
loc_str = ['A) Eastern Pacific', 'B) Central Brazil', 'C) South Atlantic', 'D) Sahara Desert', 'E) Tibetan Plateau', 'F) Western Pacific']

month = 6
years = [2018, 2019, 2020, 2021, 2022]
sats = [0, 1, 2, 3]
hdate = datetime.datetime(2020, month, 1)

# %% load data
# allocate output dicts
local_tmt = {}
local_time = {}
local_tmt_clear = {}
# loop over satellites
for sat in range(4):
    # get all data
    ltime, lat, lon, tmt, tmt_clear = get_sat_data(sat, month, years)
    # get data for each grid cell
    for ii, ploc in enumerate(plocs):
        # get index for gridcell
        llat = ploc[0]
        llon = ploc[1]
        inds = np.where((lon > llon - 0.5) & (lon <= llon + 0.5) & \
                        (lat > llat - 0.5) & (lat <= llat + 0.5))[0]
        print(sat, ii, llat, llon, len(inds), len(tmt))
        # subset all data for grid cell
        ltmt = tmt[inds]
        ltmtc = tmt_clear[inds]
        lltime = ltime[inds]
        # store data
        if ii not in local_tmt.keys():
            local_tmt[ii] = list(ltmt)
            local_tmt_clear[ii] = list(ltmtc)
            local_time[ii] = list(lltime)
        else:
            local_tmt[ii] = local_tmt[ii] + list(ltmt)
            local_tmt_clear[ii] = local_tmt_clear[ii] + list(ltmt)
            local_time[ii] = local_time[ii] + list(lltime)

# get reference data
fn = '/p/vast1/pochedls/era5/tmt/climatology/tmt_reference_monthly_diurnal_local_1x1_2013-2022.nc'
local_tmt_reference = {}
plocs = np.array(plocs)
ds = xr.open_dataset(fn)
ds = ds.sel(time=slice('2018-01', '2022-12'))
time_reference = np.arange(0, 25)
for ii in range(len(plocs)):
    tmt = ds.tmt.sel(lat=plocs[ii, 0], lon=plocs[ii, 1]).isel(time=np.arange(month-1, len(ds.time), 12))
    local_tmt_reference[ii] = tmt.values
ds.close()

# get fitted data
fn = '/p/vast1/pochedls/era5/tmt/climatology/tmt_sampled_monthly_diurnal_local_1x1_2018-2022_5fps_4sats_0.25K.nc'
local_tmt_fitted = {}
plocs = np.array(plocs)
ds = xr.open_dataset(fn)
ds = ds.sel(time=slice('2018-01', '2022-12'))
for ii in range(len(plocs)):
    tmt = ds.tmt.sel(lat=plocs[ii, 0], lon=plocs[ii, 1]).isel(time=np.arange(month-1, len(ds.time), 12))
    local_tmt_fitted[ii] = tmt.values
ds.close()

# %% make plot
plt.figure(figsize=(8, 9))
for ii in range(len(loc_str)):
    plt.subplot(3, 2, ii+1)
    # get anomalies
    local_tmt_anomaly = local_tmt[ii] - np.mean(local_tmt[ii])
    local_tmt_fitted_anomaly = local_tmt_fitted[ii] - np.expand_dims(np.mean(local_tmt_fitted[ii], axis=1), axis=1)
    local_tmt_fitted_anomaly = np.concatenate((local_tmt_fitted_anomaly, np.expand_dims(local_tmt_fitted_anomaly[:, 0], axis=1)), axis=1)
    local_tmt_reference_anomaly = local_tmt_reference[ii] - np.expand_dims(np.mean(local_tmt_reference[ii], axis=1), axis=1)
    local_tmt_reference_anomaly = np.concatenate((local_tmt_reference_anomaly, np.expand_dims(local_tmt_reference_anomaly[:, 0], axis=1)), axis=1)
    # fit data
    popt, pcov = curve_fit(fourier, local_time[ii], local_tmt_anomaly)
    dc = fourier(local_time[ii], *popt)
    # compute signal / noise / n / r
    noise = np.std(local_tmt_anomaly - dc) / 2.
    # signal = np.max(dc) - np.min(dc)
    signal = (np.max(np.mean(local_tmt_reference_anomaly, axis=0)) - np.min(np.mean(local_tmt_reference_anomaly, axis=0))) / 2.
    n = len(local_tmt_anomaly)
    r = np.corrcoef(np.mean(local_tmt_fitted_anomaly, axis=0)[0:24], np.mean(local_tmt_reference_anomaly, axis=0)[0:24])[0, 1]
    # plot measurements
    plt.plot(local_time[ii], local_tmt_anomaly, '.', markersize=3, color='gray')
    # plot fitted diurnal cycle
    plt.plot(time_reference, np.mean(local_tmt_fitted_anomaly, axis=0), 'k-', linewidth=2, zorder=260.)
    plt.fill_between(time_reference, np.min(local_tmt_fitted_anomaly, axis=0), np.max(local_tmt_fitted_anomaly, axis=0), color='k', alpha=0.7, zorder=250., linewidth=0.)
    # plot reference diurnal cycle
    plt.plot(time_reference, np.mean(local_tmt_reference_anomaly, axis=0), 'r', zorder=502, linewidth=0.5)
    plt.fill_between(time_reference, np.min(local_tmt_reference_anomaly, axis=0), np.max(local_tmt_reference_anomaly, axis=0), color='r', alpha=0.7, zorder=500., linewidth=0.)
    # update axes
    ymax = np.max(local_tmt_anomaly)
    ymin = np.min(local_tmt_anomaly)
    yamp = ymax - ymin
    plt.ylim(ymin - yamp*0.4, ymax + yamp*0.1)
    plt.xlim(0, 24)
    plt.xlabel('Local Time')
    plt.ylabel('Diurnal Anomaly [K]')
    plt.xticks([0, 6, 12, 18, 24])
    # write out stats
    Sstr = "{:.2f}".format(np.round(signal, 2))
    Nstr = "{:.2f}".format(np.round(noise, 2))
    SNstr = "{:.2f}".format(np.round(signal/noise, 2))
    plt.text(2, ymin - yamp*0.1, 'n = ' + str(n))
    plt.text(2, ymin - yamp*0.23, 'S / N = ' + Sstr + ' / ' + Nstr + ' (' + SNstr + ')')
    plt.text(2, ymin - yamp*0.36, 'r = ' + "{:.2f}".format(np.round(r, 2)))
    if ii == 0:
        plt.text(17, ymin - yamp*0.1, 'Reference', color='r')
        plt.text(17, ymin - yamp*0.23, 'Sampled', color='gray')
        plt.text(17, ymin - yamp*0.36, 'Fitted', color='k')
    # update spines
    ax = plt.gca()
    # Move left and bottom spines outward by 10 points
    ax.spines.left.set_position(('outward', 10))
    ax.spines.bottom.set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # title
    plt.title(loc_str[ii], loc='left')
plt.tight_layout()
plt.savefig('../figs/fit_examples_' + hdate.strftime("%B") + '.pdf', bbox_inches='tight')
plt.show()
