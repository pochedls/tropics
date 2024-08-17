# -*- coding: utf-8 -*-

"""plot_satellites.py
    
    Python script to plot satellite paths and sampling.

    Author: Stephen Po-Chedley
"""

# %% imports
import xarray as xr
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# %% helper functions
def get_sat_data(sat, day, span, pday):
    """
    get_sat_data(sat, day, span, pday)

    Function returns the time, lat, and lon points for a given
    satellite and time period.

    Parameters:
    -----------
    sat (int):     satellite id
    day (int):     day of year (1-365)
    span (int):    stride length of sampling
    pday (float):  fraction of day to plot

    Returns:
    --------
    List            : time values
    xr.DataArray    : latitude values
    xr.DataArray    : longitude values
    """
    daystr = str(day).zfill(3)
    satstr = str(sat).zfill(2)
    dpath = '/p/vast1/pochedls/tropics/tropics_x22_raan180_v2/'
    fn = dpath + satstr + '/' + 'tropics_x22_raan180_v2_' + satstr + '_DAY_' + daystr + '.nc'
    ds = xr.open_dataset(fn)
    time = ds.TIME
    inds = np.arange(0, int(len(time)*pday), span)
    LAT = ds.LAT.isel(LAT_dim_1=inds).isel(LAT_dim_0=39)
    LON = ds.LON.isel(LON_dim_1=inds).isel(LON_dim_0=39)
    time = time.isel(TIME_dim_0=inds)
    basetime = datetime.datetime(2022, 1, 1) + datetime.timedelta(days=day - 1)
    time = [basetime + datetime.timedelta(seconds=int(s)) for s in time.values]
    return time, LAT, LON


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


# %% parameters
dpath = '/p/vast1/pochedls/tropics/tropics_x22_raan180_v2/'  # data path
clist = ['r', 'b', 'purple', 'orange'] # sat colors
sats = [0, 1, 2, 3]
month = 1
nlon = np.arange(0.5, 360., 1.)
nlat = np.arange(-35.5, 36, 1.)

# %% calculations
# get monthly satellite data (for one hour)
day_start = (datetime.datetime(2020, month, 1) - datetime.datetime(2019, 12, 31)).days
day_end = (datetime.datetime(2020, month+1, 1) - datetime.datetime(2019, 12, 31)).days
hdate = datetime.datetime(2020, month, 1)
all_lat = []
all_lon = []
hour = [0, 24]
for day in range(day_start, day_end):
    print(day)
    for sat in range(4):
        time, lat, lon = get_sat_data(sat, day, 1, 1)
        lat = lat.values
        lon = lon.values
        ltime = np.array([get_local_time(utctime, lon) for utctime, lon in zip(time, lon)])
        inds = np.where((ltime >= hour[0]) & (ltime < hour[1]))[0]
        lat = list(lat[inds])
        lon = list(lon[inds])
        all_lat = all_lat + lat
        all_lon = all_lon + lon
all_lat = np.array(all_lat)
all_lon = np.array(all_lon)
all_lon[all_lon < 0.] = all_lon[all_lon < 0.] + 360.
# # grid bin count
# LON, LAT = np.meshgrid(nlon, nlat)
# tree = spatial.cKDTree(list(zip(LAT.flat, LON.flat)))    
# dis, ind = tree.query(np.array([all_lat, all_lon]).T)
# hour_count = np.bincount(ind,  minlength=len(LAT.flat)) # Sum of data in each grid box
# hour_count = np.reshape(hour_count, (len(nlat), len(nlon)))

# get reference diurnal amplitudes
fn = '/p/vast1/pochedls/era5/tmt/climatology/tmt_reference_monthly_diurnal_1x1_2013-2022.nc'
ds = xr.open_dataset(fn)
ds = ds.sel(time=slice('2018-01', '2022-12'))
amplitude = []
for i in range(len(ds.time)):
    tmtm = ds.tmt.isel(time=i)
    mamp = tmtm.max(dim='hour') - tmtm.min(dim='hour')
    amplitude.append(mamp)
amplitude = xr.concat(amplitude, dim=ds.time)
monthly_amplitude = xr.concat(amplitude, dim=ds.time)
monthly_amplitude = monthly_amplitude.groupby('time.month').mean()

# get samples
fn = '/p/vast1/pochedls/era5/tmt/climatology/tmt_sampled_monthly_diurnal_local_1x1_2018-2022_5fps_4sats_0.25K.nc'
ds = xr.open_mfdataset(fn)
ts_n = ds.n_samples.sum(dim='lat').sum(dim='lon')
ts_ni = ds.n_samples_ignored.sum(dim='lat').sum(dim='lon')
print('percent_samples_ignored: ' + str(np.mean(ts_ni.values / (ts_n.values + ts_ni.values) * 100.)))
n_samples = ds.n_samples[month - 1]

# %% make figure
fig = plt.figure(figsize=(5, 6))

# plot one hour trajectory all satellites
ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
to_ind = 120
for ii, sat in enumerate(sats):
    # get color
    c = clist[ii]
    time, lat, lon = get_sat_data(sat, day_start, 30, 1)
    # plot path points
    ax.plot(lon[0:to_ind], lat[0:to_ind], '.', markersize=2, color=c, transform=ccrs.PlateCarree())
    # plot trajectory arrow
    ax.arrow(lon[to_ind], lat[to_ind], lon[to_ind+1] - lon[to_ind], lat[to_ind+1] - lat[to_ind], facecolor=c, edgecolor=c, width=1.5, transform=ccrs.PlateCarree(), zorder=100)
# add coastlines
ax.coastlines()
# limit to tropics
ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
plt.title('A. Satellite Trajectories', loc='left')

# one month sampling all satellites
ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
cticks = list(np.log10(np.arange(100, 2301, 100)))
cticklabels = ['100', '200', '', '', '500', '', '', '', '', '1000', '', '', '', '', '1500', '', '', '', '', '', '', '', '2300']
clevs = np.arange(2, 3.38, 0.01)
data, lon = add_cyclic_point(np.log10(n_samples), coord=nlon)
im = ax.contourf(lon, nlat, data, clevs, cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
# add coastlines
ax.coastlines()
# limit to tropics
ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
cb = plt.colorbar(im, ticks=cticks, orientation='horizontal')
cb.ax.set_xticklabels(cticklabels)
plt.title('B. Monthly Sample Count', loc='left')
print('minimum ' + str(np.nanmin(n_samples.values)))
print('median ' + str(np.nanmedian(n_samples.values)))

# diurnal amplitude
ax = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
# fill in NaN values
data = np.log10(monthly_amplitude[month])
data = data.sel(lat=slice(-36, 36))
data = xr.where(~np.isnan(n_samples), data, np.nan)
# data = xr.where(n_samples != 0., data, np.nan)
cticks = list(np.log10(np.arange(0.1, 1.1, 0.1))) + list(np.log10(np.arange(2, 10.1, 1)))
cticklabels = ['0.1', '', '', '', '', '', '', '', '', '1'] + ['', '', '', '', '', '', '', '', '10']
clevs = np.arange(-1, 1.01, 0.05)
pdata, lon = add_cyclic_point(data, coord=nlon)
im = ax.contourf(lon, data.lat, pdata, clevs, cmap=plt.cm.viridis, transform=ccrs.PlateCarree(), extend='both')
# add coastlines
ax.coastlines()
# limit to tropics
ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
cb = plt.colorbar(im, ticks=cticks, orientation='horizontal')
cb.ax.set_xticklabels(cticklabels)
# East Pacific, Amazon, South Atlantic, Sahara, Himalayas, West Pacific
plocs = [[-8.5, 242.5],
         [-11.5, 308.5],
         [-23.5, 344.5],
         [24.5, 17.5],
         [29.5, 90.5],
         [-29.5, 176.5]]
plocs = np.array(plocs)
plt.scatter(x=plocs[:, 1], y=plocs[:, 0],
            color="red",
            s=8,
            alpha=0.7,
            transform=ccrs.PlateCarree()) ## Important
plt.title('C. Diurnal Amplitude', loc='left')
plt.tight_layout()
plt.savefig('../figs/sampling_' + hdate.strftime("%B") + '.pdf', bbox_inches='tight')
plt.show()




