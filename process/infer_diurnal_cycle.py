import xarray as xr
import numpy as np
import glob
import datetime
from scipy import spatial
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import argparse


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


def fit_diurnal_index(gridind, ind, scanbt, scanbtc, time, bt_cutoff):
    """
    fit_diurnal_index(gridind, ind, scanbt, scanbtc, time, bt_cutoff)

    Parameters:
    -----------
    gridind (int)   : grid index to fit
    ind (Array)     : array of index mappings to target grid
    scanbt          : array of tmt brightness temperatures
    scanbtc         : array of clear-sky tmt brightness temperatures
    time            : array of measurement times
    bt_cutoff       : max value in which tmt can differ from tmt_clear

    Returns:
    --------
    dc (array)                  : diurnal cycle of grid cell [at hours 0, 1, 2, ..., 23]
    popt   (array-like)         : fit values for diurnal cycle at grid cell
    dc_clear (array)            : clear-sky diurnal cycle of grid cell [at hours 0, 1, 2, ..., 23]
    popt_clear (array-like)     : clear-sky fit values for diurnal cycle at grid cell
    n (int)                     : number of points used in fit
    nbad (int)                  : number of points ignored in fit
    """
    # create reference time vector [0, 1, 2, ..., 23]
    time_reference = np.arange(0, 24)
    # get tmt values for grid cell
    bt = scanbt[ind == gridind]
    btc = scanbtc[ind == gridind]
    ltime = time[ind == gridind]
    # get total points
    n = len(bt)
    # get good points
    if bt_cutoff > 0:
        igood = np.where(np.abs(bt - btc) < bt_cutoff)[0]
    else:
        igood = np.arange(len(bt))
    # get sample points
    ns = len(igood)
    # get bad points
    nbad = n - ns
    bt = bt[igood]
    btc = btc[igood]
    ltime = ltime[igood]
    # full sky curve fit
    popt, pcov = curve_fit(fourier, ltime, bt)
    dc = fourier(time_reference, *popt)
    # # clear sky curve fit
    popt_c, pcov_c = curve_fit(fourier, ltime, btc)
    dc_c = fourier(time_reference, *popt_c)
    return dc, dc_c, popt, popt_c, ns, nbad


def get_nn_mapping(nlat, nlon, scanlat, scanlon):
    """
    get_nn_mapping(nlat, nlon, scanlat, scanlon)

    Function determines the nearest neighbor grid cell for
    an array of satellite measurement locations (scanlat and
    scanlon). The grid cells are determined from a latitude
    and longitude array, which together form [nlat, nlon].

    Parameters:
    -----------
    nlat (Array)       : array of latitude grid points to map to
    nlon (Array)       : array of longitude grid points to map to
    scanlat (List)     : list of scan latitude points
    scanlon (List)     : list of scan longitude points

    Returns:
    --------
    indset : set of all index locations for which there is data
    ind    : indices mapping to each grid cell [nlat*nlon]
    """
    # construct mesh grid
    LON, LAT = np.meshgrid(nlon, nlat)
    # construct KD Tree
    tree = spatial.cKDTree(list(zip(LAT.flat, LON.flat)))    
    # get indices for nearest points
    dis, ind = tree.query(np.array([scanlat, scanlon]).T)
    # get indices with values
    indset = list(set(ind))
    return indset, ind


def load_file(fn, fprints):
    """
    load_file(fn, fprints)

    Function reads in one file for full and clear-sky TMT
    data for a subset of footprints.

    Parameters:
    -----------
    fn (str)                : file to read
    fprints (array-like)    : footprint indices to read in

    Returns:
    --------
    time (List)        : list of local time (in hours)
    scanlat (List)     : list of scan latitude points
    scanlon (List)     : list of scan longitude points
    scanbt (List)      : list of tmt brightness temperatures
    scanbtc (List)     : list of clear-sky tmt brightness temperature values
    """
    # declare variable ids
    vid = 'tmt'
    vid_clear = 'tmt_clear'
    ds = xr.open_dataset(fn, use_cftime=True)
    ds = ds.isel(footprint=fprints)
    ds.load()
    # reshape time
    timef = ds.time
    timef = np.tile(np.expand_dims(timef, axis=1), (1, len(fprints)))
    # flatten
    timef = np.reshape(np.array(timef), -1)
    btf = np.reshape(np.array(ds[vid]), -1)
    btcf = np.reshape(np.array(ds[vid_clear]), -1)
    latf = np.reshape(np.array(ds.scanlat), -1)
    lonf = np.reshape(np.array(ds.scanlon), -1)
    timef = [get_local_time(timef[i], lonf[i]) for i in range(len(timef))]
    # to list
    time = timef
    scanbt = list(btf)
    scanbtc = list(btcf)
    scanlon = list(lonf)
    scanlat = list(latf)
    # clean up
    ds.close()
    return time, scanlat, scanlon, scanbt, scanbtc


def load_data(files, fprints, n_jobs=10):
    """
    load_data(files, fprints)

    Function reads in a list of files (in parallel) for full and 
    clear-sky TMT data for a subset of footprints.

    Parameters:
    -----------
    files (list)                : list of files to read
    fprints (array-like)        : footprint indices to read in
    n_jobs (int)                : number of parallel workers

    Returns:
    --------
    time (array)        : array of local time (in hours)
    scanlat (array)     : array of scan latitude points
    scanlon (array)     : array of scan longitude points
    scanbt (array)      : array of tmt brightness temperatures
    scanbtc (array)     : array of clear-sky tmt brightness temperature values
    """
    # declare variable ids
    results = Parallel(n_jobs=n_jobs)(delayed(load_file)(fn, fprints) for fn in files)
    time = scanlat = scanlon = scanbt = scanbtc = []
    for row in results:
        time = time + row[0]
        scanlat = scanlat + row[1]
        scanlon = scanlon + row[2]
        scanbt = scanbt + row[3]
        scanbtc = scanbtc + row[4]
    time = np.array(time)
    scanlat = np.array(scanlat)
    scanlon = np.array(scanlon)
    scanbt = np.array(scanbt)
    scanbtc = np.array(scanbtc)
    return time, scanlat, scanlon, scanbt, scanbtc


def save_data(fnOut, nlat, nlon, mtimes, diurnal_cycle, diurnal_fit, diurnal_cycle_clear, diurnal_fit_clear, n_samples, n_bad_samples):
    """
    save_data(fnOut, nlat, nlon, mtimes, diurnal_cycle, diurnal_fit, diurnal_cycle_clear, diurnal_fit_clear, n_samples, n_bad_samples)

    Function takes in fitted diurnal cycle data and saves it to a file.

    Parameters:
    -----------
    fnOut (str)                     : filename to save data
    nlat (array)                    : latitude array of output grid
    nlon (array)                    : longitude array of output grid
    mtimes (array)                  : array of time values (datetime objects)
    diurnal_cycle (array)           : diurnal cycle of TMT [time, hour, lat, lon]
    diurnal_fit (array-like)        : coefficient values of diurnal cycle [time, coef, lat, lon]
    diurnal_cycle_clear (array)     : diurnal cycle of clear-sky TMT [time, hour, lat, lon]
    diurnal_fit_clear (array-like)  : coefficient values of clear-sky diurnal cycle [time, coef, lat, lon]
    n_samples (array)               : number of samples used in fit [time, lat, lon]
    n_bad_samples (array)           : number of samples ignored in fit [time, lat, lon]
    """
    lat = xr.DataArray(
        data=nlat,
        name='lat',
        dims=["lat"],
        coords=dict(lat=(["lat"], nlat)),
        attrs=dict(
            long_name="latitude",
            units="degrees_north"))

    lon = xr.DataArray(
        data=nlon,
        name='lon',
        dims=["lon"],
        coords=dict(lon=(["lon"], nlon)),
        attrs=dict(
            long_name="longitude",
            units="degrees_east"))

    time = xr.DataArray(
        data=mtimes,
        name='time',
        dims=["time"],
        coords=dict(time=(["time"], mtimes)),
        attrs=dict(
            long_name="time"))

    hour = xr.DataArray(
        data=np.arange(0, 24),
        name='hour',
        dims=["hour"],
        coords=dict(hour=(["hour"], np.arange(0, 24))),
        attrs=dict(
            long_name="hour"))

    coef_index = xr.DataArray(
        data=np.arange(5),
        name='coef_index',
        dims=["coef_index"],
        coords=dict(coef_index=(["coef_index"], np.arange(5))),
        attrs=dict(
            long_name="coefficent_index",
            description="a0, a1, a2, t1, t2"))

    dc = xr.DataArray(
        data=diurnal_cycle,
        name='tmt',
        dims=["time", "hour", "lat", "lon"],
        coords=dict(
            time=time,
            hour=hour,
            lat=lat,
            lon=lon,
        ),
        attrs=dict(
            description="TMT Diurnal Cycle",
            units="K",
        ),
    )

    dcc = xr.DataArray(
        data=diurnal_cycle_clear,
        name='tmt_clear',
        dims=["time", "hour", "lat", "lon"],
        coords=dict(
            time=time,
            hour=hour,
            lat=lat,
            lon=lon,
        ),
        attrs=dict(
            description="TMT Clear-sky Diurnal Cycle",
            units="K",
        ),
    )

    ns = xr.DataArray(
        data=n_samples,
        name='n_samples',
        dims=["time", "lat", "lon"],
        coords=dict(
            time=time,
            lat=lat,
            lon=lon,
        ),
        attrs=dict(
            description="Samples",
            units="1",
        ),
    )

    nsb = xr.DataArray(
        data=n_bad_samples,
        name='n_samples_ignored',
        dims=["time", "lat", "lon"],
        coords=dict(
            time=time,
            lat=lat,
            lon=lon,
        ),
        attrs=dict(
            description="Samples ignored",
            units="1",
        ),
    )

    dfit = xr.DataArray(
        data=diurnal_fit,
        name='coef',
        dims=["time", "coef_index", "lat", "lon"],
        coords=dict(
            time=time,
            coef_index=coef_index,
            lat=lat,
            lon=lon,
        ),
        attrs=dict(
            description="Diurnal fit"
        ),
    )

    dfitc = xr.DataArray(
        data=diurnal_fit_clear,
        name='coef_clear',
        dims=["time", "coef_index", "lat", "lon"],
        coords=dict(
            time=time,
            coef_index=coef_index,
            lat=lat,
            lon=lon,
        ),
        attrs=dict(
            description="Diurnal fit clear-sky"
        ),
    )

    dc = dc.to_dataset()
    dc['tmt_clear'] = dcc
    dc['coef'] = dfit
    dc['coef_clear'] = dfitc
    dc['n_samples'] = ns
    dc['n_samples_ignored'] = nsb
    # output to netcdf
    dc.to_netcdf(fnOut)



# specify argparse arguments
parser = argparse.ArgumentParser(description='Process arguments for infer_diurnal_cycle.py.')
parser.add_argument('compset', metavar='c', type=str,
                    help='Component set string to use')
args = parser.parse_args()

# parameters
years = np.arange(2018, 2023) # years to analyze
dpath = '/p/vast1/pochedls/era5/tropics/' # input data directory
dpathout = '/p/vast1/pochedls/era5/tmt/climatology/' # output data directory
desc = args.compset
# desc = 'nadir_4sats_all'

# define parameter sets
compset = {'nadir_4sats_0.25K': {'sats': np.arange(0, 4), 'fprints': [39], 'bt_cutoff': 0.25},
           'nadir_4sats_all': {'sats': np.arange(0, 4), 'fprints': [39], 'bt_cutoff': -1},
           '5fps_4sats_0.25K': {'sats': np.arange(0, 4), 'fprints': [19, 29, 39, 49, 59], 'bt_cutoff': 0.25},
           '5fps_4sats_all': {'sats': np.arange(0, 4), 'fprints': [19, 29, 39, 49, 59], 'bt_cutoff': -1},
           'nadir_6sats_0.25K': {'sats': np.arange(0, 6), 'fprints': [39], 'bt_cutoff': 0.25},
           '5fps_6sats_0.25K': {'sats': np.arange(0, 6), 'fprints': [19, 29, 39, 49, 59], 'bt_cutoff': 0.25}}
bt_cutoff = compset[desc]['bt_cutoff']
fprints = compset[desc]['fprints']
sats = compset[desc]['sats']

# define output grid
grid_label = '1x1'  # 2.5x2.5
nlon = np.arange(0.5, 360., 1.)  # np.arange(1.25, 360., 2.5)
nlat = np.arange(-35.5, 36, 1.)  # np.arange(-41.25, 42, 2.5)

# start computation
print('Starting job: ' + desc)
# pre-allocate output grids
diurnal_fit = np.zeros((len(years)*12, 5, len(nlat), len(nlon))) * np.nan
diurnal_cycle = np.zeros((len(years)*12, 24, len(nlat), len(nlon))) * np.nan
diurnal_fit_clear = np.zeros((len(years)*12, 5, len(nlat), len(nlon))) * np.nan
diurnal_cycle_clear = np.zeros((len(years)*12, 24, len(nlat), len(nlon))) * np.nan
n_samples = np.zeros((len(years)*12, len(nlat), len(nlon))) * np.nan
n_bad_samples = np.zeros((len(years)*12, len(nlat), len(nlon))) * np.nan
mtimes = []

# loop over years and months
for iyear, year in enumerate(years):
    for im in range(12):
        # progress update
        print(datetime.datetime.now(), year, im+1)
        # append year / month timestamp to output time array
        mtimes.append(datetime.datetime(year, im + 1, 15))
        # get files of interest
        files = [glob.glob(dpath + 'tmt_daily_sat' + str(sat).zfill(2) + '_' + str(year) + str(im+1).zfill(2) + '*.nc') for sat in sats]
        files = [fn for slist in files for fn in slist]
        # load data
        time, scanlat, scanlon, scanbt, scanbtc = load_data(files, fprints)
        # get nearest neighbor mapping
        indset, ind = get_nn_mapping(nlat, nlon, scanlat, scanlon)
        # solve for each grid cell's diurnal cycle (in parallel)
        results = Parallel(n_jobs=36)(delayed(fit_diurnal_index)(gridind, ind, scanbt, scanbtc, time, bt_cutoff) for gridind in indset)
        # construct monthly output grids
        month_dc = np.zeros((24, len(nlat)*len(nlon))) * np.nan
        month_dcf = np.zeros((5, len(nlat)*len(nlon))) * np.nan
        month_dcc = np.zeros((24, len(nlat)*len(nlon)))  * np.nan
        month_dcfc = np.zeros((5, len(nlat)*len(nlon))) * np.nan
        month_ns = np.zeros((len(nlat)*len(nlon))) * np.nan
        month_nbs = np.zeros((len(nlat)*len(nlon))) * np.nan
        # put results into monthly output grids
        for i, row in enumerate(results):
            dc, dc_c, popt, popt_c, n, nbad = row
            month_dc[:, indset[i]] = dc
            month_dcf[:, indset[i]] = popt
            month_dcc[:, indset[i]] = dc_c
            month_dcfc[:, indset[i]] = popt_c
            month_ns[indset[i]] = n
            month_nbs[indset[i]] = nbad
        # put data into output grids
        itime = (iyear*12) + im # time index
        diurnal_cycle[itime, :, :, :] = np.reshape(month_dc, (24, len(nlat), len(nlon)))
        diurnal_fit[itime, :, :, :] = np.reshape(month_dcf, (5, len(nlat), len(nlon)))
        diurnal_cycle_clear[itime, :, :, :] = np.reshape(month_dcc, (24, len(nlat), len(nlon)))
        diurnal_fit_clear[itime, :, :, :] = np.reshape(month_dcfc, (5, len(nlat), len(nlon)))
        n_samples[itime, :, :] = np.reshape(month_ns, (len(nlat), len(nlon)))
        n_bad_samples[itime, :, :] = np.reshape(month_nbs, (len(nlat), len(nlon)))

# create output file name
fnOut = dpathout + 'tmt_sampled_monthly_diurnal_local_' + grid_label + '_' + str(years[0]) + '-' + str(years[-1]) + '_' + desc + '.nc'
# save data
save_data(fnOut, nlat, nlon, mtimes, diurnal_cycle, diurnal_fit, diurnal_cycle_clear, diurnal_fit_clear, n_samples, n_bad_samples)
