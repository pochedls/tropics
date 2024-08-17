# -*- coding: utf-8 -*-

"""plot_satellites.py
    
    Python script to plot satellite paths.

    Author: Stephen Po-Chedley
"""

# %% imports
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs

# %% parameters
dpath = '/p/vast1/pochedls/tropics/tropics_x22_raan180_v2/'  # data path
span = 30  # stride length (30 is one point per minute)
pday = 1 # portion of day to plot (0 - 1)
clist = ['r', 'b', 'purple', 'orange', 'k', 'cyan'] # sat colors
to_ind = 60 # index to plot to

# %%
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
    fn = dpath + satstr + '/' + 'tropics_x22_raan180_v2_' + satstr + '_DAY_' + daystr + '.nc'
    ds = xr.open_dataset(fn)
    time = ds.TIME
    inds = np.arange(0, int(len(time)*pday), span)
    LAT = ds.LAT.isel(LAT_dim_1=inds).isel(LAT_dim_0=41)
    LON = ds.LON.isel(LON_dim_1=inds).isel(LON_dim_0=41)
    time = time.isel(TIME_dim_0=inds)
    basetime = datetime.datetime(2022, 1, 1) + datetime.timedelta(days=day - 1)
    time = [basetime + datetime.timedelta(seconds=int(s)) for s in time.values]
    return time, LAT, LON



day = 1

# %% get sat paths
# loop over satellites and get data
sat_data = []
for sat in [0, 1, 2, 3]:
    time, lat, lon = get_sat_data(sat, day, span, pday)
    sat_data.append([time, lat, lon])
# add a satellite 180 degrees ahead of 00
time, lat, lon = sat_data[0]
lon = lon + 180
lon[lon > 180] = lon[lon > 180] - 360
sat_data.append([time, lat, lon])
# add a satellite 180 degrees ahead of 02
time, lat, lon = sat_data[2]
lon = lon + 180
lon[lon > 180] = lon[lon > 180] - 360
sat_data.append([time, lat, lon])

# %% plot figure
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# get time of last point
date = time[to_ind]
# loop over satellites and plot tracks
for ii, row in enumerate(sat_data):
    # get track data
    time, LAT, LON = row
    # get color
    c = clist[ii]
    # plot path points
    ax.plot(LON[0:to_ind+1], LAT[0:to_ind+1], '.', color=c, transform=ccrs.PlateCarree())
    # plot trajectory arrow
    ax.arrow(LON[to_ind], LAT[to_ind], LON[to_ind+1] - LON[to_ind], LAT[to_ind+1] - LAT[to_ind], facecolor=c, edgecolor=None, width=1.5, transform=ccrs.PlateCarree(), zorder=100)
    # print last sat point
    print(ii, c, LAT.values[to_ind], LON.values[to_ind])
# add coastlines
ax.coastlines()
# limit to tropics
ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
plt.show()
