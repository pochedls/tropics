# %% imports
import xcdat as xc
import numpy as np
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from cartopy.util import add_cyclic_point


def plot_map(bt, date, clevs, title=None, nightshade=True, textlabel=None, cmap=plt.cm.RdBu_r, extend='both', plot_sun=False):
    lat, lon = bt.lat, bt.lon
    bmap, clon = add_cyclic_point(bt, coord=lon)
    date = datetime.datetime(date.year, date.month, date.day, date.hour)
    im = plt.contourf(clon, lat, bmap, clevs, cmap=cmap, transform=ccrs.PlateCarree(), extend=extend)
    # plot solar noon
    geodetic = ccrs.Geodetic()
    platecarree = ccrs.PlateCarree()
    slon = np.mod(360-(date.hour + 12)*15, 360)
    ad_lon_t, ad_lat_t = platecarree.transform_point(slon-1, 90, geodetic)
    if plot_sun:
        liv_lon_t, liv_lat_t = platecarree.transform_point(slon-1, -90, geodetic)
        plt.plot([ad_lon_t, liv_lon_t], [ad_lat_t, liv_lat_t], color='red', linewidth=0.5, transform=platecarree)
    ax = plt.gca()
    if textlabel is not None:
        plt.text(185, -25, textlabel, transform=platecarree, fontsize=10)
    if title is not None:
        ax.set_title(title, loc='left')
    if nightshade:
        ax.add_feature(Nightshade(date, alpha=0.2))
    ax.coastlines()
    return im


# %% parameters
dpath = '/p/vast1/pochedls/era5/tmt/climatology/'
fn_ref = dpath + 'tmt_reference_monthly_diurnal_local_1x1_2013-2022.nc'
compset = '5fps_4sats_all'
fn_test = dpath + 'tmt_sampled_monthly_diurnal_local_1x1_2018-2022_' + compset + '.nc'
time_interval = slice("2018-01-01", "2022-12-31")
phours = [2, 6, 10, 14, 18, 22]
month = 6

# %% open climatologies
dsr = xc.open_dataset(fn_ref)
dst = xc.open_dataset(fn_test)
# subset in time
dsr = dsr.sel(time=time_interval)
dst = dst.sel(time=time_interval)
# to same latitude range
dsr = dsr.sel(lat=slice(-28.75, 28.75))
dst = dst.sel(lat=slice(-28.75, 28.75))
# get monthly data
dsr = dsr.groupby("time.month").mean(dim='time')
dst = dst.groupby("time.month").mean(dim='time')
dsr = dsr.sel(month=month).load()
dst = dst.sel(month=month).load()
# diurnal anomalies
tmtr = dsr.tmt - dsr.tmt.mean(dim='hour')
tmtt = dst.tmt - dsr.tmt.mean(dim='hour')

# %% make stacked plots
clevsd = np.arange(-0.5, 0.51, 0.05)
clevsa = np.arange(-2.5, 2.51, 0.25)
hours = [0, 4, 8, 12, 16, 20]
fig = plt.figure(figsize=(8, 5))
cmap = plt.cm.RdBu_r
for i, hour in enumerate(hours):
    print(hour)
    plt.subplot(len(hours), 3, (i*3)+1, projection=ccrs.PlateCarree())
    tmt_test = tmtt.isel(hour=hour)
    tmt_ref = tmtr.isel(hour=hour)
    hrstr = str(hour).zfill(2) + ':00'
    hdate = datetime.datetime(2020, month, 15, hour)
    if i == 0:
        title1 = 'Reference'
        title2 = 'TROPICS'
        title3 = 'Difference'
    else:
        title1 = title2 = title3 = None
    im1 = plot_map(tmt_ref, hdate, clevsa, title=title1, textlabel=hrstr, cmap=cmap)
    plt.subplot(len(hours), 3, (i*3)+2, projection=ccrs.PlateCarree())
    plot_map(tmt_test, hdate, clevsa, title=title2, cmap=cmap)
    plt.subplot(len(hours), 3, (i*3)+3, projection=ccrs.PlateCarree())
    im2 = plot_map(tmt_test - tmt_ref, hdate, clevsd, title=title3, cmap=cmap)
    ax = plt.gca()
    ax.set_extent([-180, 180, -28, 28], crs=ccrs.PlateCarree())
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.025, 0.1, 0.62, 0.03])
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', ticks=np.arange(-2.5, 3, 0.5))
cbar_ax = fig.add_axes([0.675, 0.1, 0.3, 0.03])
fig.colorbar(im2, cax=cbar_ax, orientation='horizontal', ticks=np.arange(-0.5, 0.51, 0.25))
plt.savefig('../figs/maps/mapped_difference_' + hdate.strftime("%B") + '_' + compset + '.png', bbox_inches='tight')
plt.show()




