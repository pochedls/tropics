# -*- coding: utf-8 -*-

"""plot_satellites.py
    
    Python script to plot statistics from diurnal fits.

    Author: Stephen Po-Chedley
"""

# %% imports
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# %% parameters
month = 12
hdate = datetime.datetime(2020, month, 1)

# %% load data
# get reference data
fn = '/p/vast1/pochedls/era5/tmt/climatology/tmt_reference_monthly_diurnal_local_1x1_2013-2022.nc'
local_tmt_reference = {}
ds = xr.open_dataset(fn)
ds = ds.sel(time=slice('2018-01', '2022-12'))
ds = ds.sel(lat=slice(-35.5, 36))
time_reference = np.arange(0, 25)
tmt_reference = ds.tmt.isel(time=np.arange(month-1, len(ds.time), 12)).mean(dim='time')
ds.close()
# diurnal amplitude
amplitude = []
for i in range(len(ds.time)):
    tmtm = ds.tmt.isel(time=i)
    mamp = tmtm.max(dim='hour') - tmtm.min(dim='hour')
    amplitude.append(mamp)
amplitude = xr.concat(amplitude, dim=ds.time)
monthly_amplitude = xr.concat(amplitude, dim=ds.time)
monthly_amplitude = monthly_amplitude.groupby('time.month').mean()

# get fitted data
fn = '/p/vast1/pochedls/era5/tmt/climatology/tmt_sampled_monthly_diurnal_local_1x1_2018-2022_5fps_4sats_0.25K.nc'
local_tmt_fitted = {}
ds = xr.open_dataset(fn)
ds = ds.sel(time=slice('2018-01', '2022-12'))
tmt_fitted = ds.tmt.isel(time=np.arange(month-1, len(ds.time), 12)).mean(dim='time')
ds.close()

# %% get amplitude and correlation
amplitude_perror = np.zeros((len(tmt_fitted.lat), len(tmt_fitted.lon))) * np.nan
amplitude_error = np.zeros((len(tmt_fitted.lat), len(tmt_fitted.lon))) * np.nan
fit_correlation = np.zeros((len(tmt_fitted.lat), len(tmt_fitted.lon))) * np.nan
for ilat in range(len(tmt_reference.lat)):
    for ilon in range(len(tmt_reference.lon)):
        tmtrg = tmt_reference[:, ilat, ilon].values
        tmtfg = tmt_fitted[:, ilat, ilon].values
        r1 = np.corrcoef(tmtrg, tmtfg)[0, 1]
        ampr = (np.max(tmtrg) - np.min(tmtrg)) / 2.
        ampf = (np.max(tmtfg) - np.min(tmtfg)) / 2.
        amppe = (ampf - ampr) / ampr * 100.
        ampe = ampf - ampr
        amplitude_perror[ilat, ilon] = amppe
        amplitude_error[ilat, ilon] = ampe
        fit_correlation[ilat, ilon] = r1

# %% make figure
fig = plt.figure(figsize=(8, 9))
nlon = tmt_fitted.lon
nlat = tmt_fitted.lat

ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
clevs = np.arange(-1., 1.01, 0.1)
data, lon = add_cyclic_point(fit_correlation, coord=nlon)
im = ax.contourf(lon, nlat, data, clevs, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
data, lon = add_cyclic_point(monthly_amplitude[month-1], coord=nlon)
CS = ax.contour(lon, nlat, data, levels=[0.2, 1, 4], colors=['k', 'k', 'k',], transform=ccrs.PlateCarree())
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
# add coastlines
ax.coastlines()
# limit to tropics
ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
cb = plt.colorbar(im, ticks=np.arange(-1, 1.1, 0.2), orientation='horizontal')
plt.title('A) Correlation', loc='left')
# 2
ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
clevs = np.arange(-1., 1.01, 0.1)
data, lon = add_cyclic_point(amplitude_error, coord=nlon)
im = ax.contourf(lon, nlat, data, clevs, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extend='both')
data, lon = add_cyclic_point(monthly_amplitude[month-1], coord=nlon)
CS = ax.contour(lon, nlat, data, levels=[0.2, 1, 4], colors=['k', 'k', 'k',], transform=ccrs.PlateCarree())
ax.clabel(CS, CS.levels, inline=True, fontsize=10)

# add coastlines
ax.coastlines()
# limit to tropics
ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
cb = plt.colorbar(im, ticks=np.arange(-1, 1.1, 0.2), orientation='horizontal')
cb.ax.set_xlabel('[K]')
plt.title('B) Amplitude Bias', loc='left')
# 3
ax = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
clevs = np.arange(-100., 100.01, 10.)
data, lon = add_cyclic_point(amplitude_perror, coord=nlon)
im = ax.contourf(lon, nlat, data, clevs, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extend='both')
data, lon = add_cyclic_point(monthly_amplitude[month-1], coord=nlon)
CS = ax.contour(lon, nlat, data, levels=[0.2, 1, 4], colors=['k', 'k', 'k',], transform=ccrs.PlateCarree())
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
# add coastlines
ax.coastlines()
# limit to tropics
ax.set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
cb = plt.colorbar(im, ticks=np.arange(-100., 100.1, 20), orientation='horizontal')
cb.ax.set_xlabel('[%]')
plt.title('C) Percent Amplitude Bias', loc='left')
plt.tight_layout()
plt.savefig('../figs/fit_stats_' + hdate.strftime("%B") + '.png', bbox_inches='tight')
plt.show()
