import xarray as xr
import metpy
import numpy as np
import scipy
import os
import sys
sys.path.append('/usr/workspace/pochedls/rttov/wrapper/')
import pyrttov

def get_era5_file(vid, year, month, grid='1x1', dpath='/p/vast1/pochedls/era5/era5_hourly/'):
    """
    get_era5_file(vid, year, month, grid='1x1', dpath='/p/vast1/pochedls/era5/era5_hourly/')

    Function constructs the string filename for ERA5 data corresponding to a given variable id
    (vid), year, and month. This method is specific to this project, which assumes ERA5 data
    is downloaded for a single variable, month, and grid resolution at a time. An example
    filename is:

        /p/vast1/pochedls/era5/era5_hourly/t_hourly_1x1_202101.nc

    Parameters:
    -----------
    vid (str): variable id of interest (e.g., 't', 'q', 'cc', 'sp', etc.)
    year (int): year of interest
    month (int): month of interest
    grid (str, Optional): grid resolution (e.g., '0.25x0.25', '0.5x0.5', '1x1')
    dpath (str, Optional): directory in which ERA5 data is located
                           (default: /p/vast1/pochedls/era5/era5_hourly/)

    Returns:
    --------
    filename (str): constructed string filename

    """
    # year / month to string
    year = str(year)
    month = str(month).zfill(2)
    # assemble file name
    fn = dpath + '/' + vid + '_hourly_' + grid + '_' + year + month + '.nc'
    fn = fn.replace('//', '/')
    return fn


def get_cube(dtime, grid='1x1', dpath='/p/vast1/pochedls/era5/era5_hourly/'):
    """
    get_cube(dtime, grid='1x1', dpath='/p/vast1/pochedls/era5/era5_hourly/')

    Function loads a single timestep of ERA5 fields required for RTTOV calculations and returns
    the data in a single xarray.Dataset object.

    Fields include: t, cc, ciwc, clwc, crwc, cswc, q, cl, sp, skt, siconc, t2m, d2m/q2m, u10,
                    v10, snowc, sftlf, orog.

    Parameters:
    -----------
    dtime (datetime object): datetime object corresponding to year / month of interest
    grid (str, Optional): grid resolution (e.g., '0.25x0.25', '0.5x0.5', '1x1')
    dpath (str, Optional): directory in which ERA5 data is located
                           (default: /p/vast1/pochedls/era5/era5_hourly/)

    Returns:
    --------
    xr.Dataset object with ERA5 fields needed for RTTOV

    Notes:
    ------
    - 2m specific humidity is calculated from 2m dewpoint and pressure.
    - Constants (sftlf/orog) are loaded or calculated via `get_constants` for the specified grid


    """
    # define required fields
    era5_vids = ['t', 'cc', 'ciwc', 'clwc', 'crwc', 'cswc', 'q', 'cl', 'sp',
                 'skt', 'siconc', 't2m', 'd2m', 'u10', 'v10', 'snowc']
    # initialize output dataset
    ds_cube = None
    # loop over variables to load
    for vid in era5_vids:
        # get filename for variable
        fn = get_era5_file(vid, dtime.year, dtime.month)
        # open dataset
        ds = xr.open_dataset(fn)
        # get timestep for field
        field = ds[vid].sel(time=dtime).load()
        # if dataset isn't created, create it with first dataarray field
        # otherwise add dataarray to existing dataset
        if ds_cube is None:
            ds_cube = field.to_dataset()
        else:
            ds_cube[vid] = field
        # clean up
        ds.close()
    # 2m specific humidity (q2m) needs to be calculated from 2m dewpoint
    # this calculation is performed here with metpy
    clist = [v for v in ds_cube.variables]
    if 'q2m' not in clist:
        if (('t2m' in clist) & ('sp' in clist)):
            ds_cube['q2m'] = metpy.calc.specific_humidity_from_dewpoint(ds_cube.sp, ds_cube.d2m)
            ds_cube['q2m'] = ds_cube['q2m'].assign_attrs({'units': 'kg kg**-1'})
    # load constant fields into dataset (sftlf and orog)
    ds_cube = get_constants(ds_cube)
    # return dataset
    return ds_cube

def validate_cube(ds):
    """
    validate_cube(ds)

    Function converts ERA5 data into units expected by RTTOV (Pa -> hPa and % -> fraction) and
    then iterates over a unit map to ensure that the fields in the dataset match the specified
    unit mapping (see unit map specifications below).

    Parameters:
    -----------
    xr.Dataset object with ERA5 fields needed for RTTOV

    Returns:
    --------
    xr.Dataset object with fields in corrected/validated units

    Raises:
    -------
    ValueError if units are not correct

    Note:
    -----
    The expected units are:
            * m/s: u10, v10
            * K: t, skt, t2m, d2m
            * hPa: level, sp
            * kg/kg: q2m, ciwc, clwc, crwc, cswc, q
            * km: orog
            * fraction: snowc, cc, cl, siconc, sftlf
            * latitude: degrees_north
            * longitude: degrees_east
    """
    # expected units
    unitmap = {'u10': ['m s**-1', 'm/s'], 
               'v10': ['m s**-1', 'm/s'],
               't': ['K'],
               'skt': ['K'],
               't2m': ['K'],
               'd2m': ['K'],
               'level': ['hPa', 'millibars'],
               'sp': ['hPa', 'millibars'],
               'q2m': ['kg kg**-1', 'kg/kg'],
               'ciwc': ['kg kg**-1', 'kg/kg'],
               'clwc': ['kg kg**-1', 'kg/kg'],
               'crwc': ['kg kg**-1', 'kg/kg'],
               'cswc': ['kg kg**-1', 'kg/kg'],
               'q': ['kg kg**-1', 'kg/kg'],
               'latitude': ['degrees_north'],
               'longitude': ['degrees_east'],
               'orog': ['km', 'kilometers']}
    # check / change surface pressure
    if ds['sp'].units == 'Pa':
        sp = ds['sp']
        sp[:] = sp[:] / 100.
        sp = sp.assign_attrs({'units': 'hPa'})
        ds['sp'] = sp
    # winds cannot exceed 100 m/s
    v10max = np.max(np.abs(ds['v10']).values)
    u10max = np.max(np.abs(ds['u10']).values)
    # make sure total wind speed (u**2 + v**2) <= 100.
    # by re-scaling the component over 100m/s so that the total
    # is less than 100.
    if u10max > 100.:
        ds['u10'] = xr.where(np.abs(ds['u10']) <= 100., ds['u10'], np.sign(ds['u10']) * np.sqrt(99**2 - ds['v10']**2), keep_attrs=True)
    if v10max > 100.:
        ds['v10'] = xr.where(np.abs(ds['v10']) <= 100., ds['v10'], np.sign(ds['v10']) * np.sqrt(99**2 - ds['u10']**2), keep_attrs=True)
    # check / change snow cover
    if ds['snowc'].units == '%':
        snowc = ds['snowc']
        snowc[:] = snowc[:] / 100.
        snowc = snowc.assign_attrs({'units': 'fraction'})
        ds['snowc'] = snowc
    # humidity minimum value is 0.1E-10
    ds['q'] = xr.where(ds['q'] > 0.1E-10, ds['q'], 0.1E-10, keep_attrs=True)
    # area coverage in fraction (0-1)
    for v in ['cc', 'cl', 'siconc', 'sftlf', 'snowc']:
        if float(ds[v].max().values) > 1:
            raise ValueError('Expected ' + v + ' to be a fraction.)')
        if float(ds[v].min().values) < 0:
            raise ValueError('Unexpected fraction below zero for: ' + v + '.')
    ## check expected units
    # vertical fields ordered from low-to-high pressure
    # wind in m/s
    # temperature in K
    # hydrologic fields in kg/kg
    for v in unitmap.keys():
        if ds[v].units not in unitmap[v]:
            raise ValueError('Units not correct for ' + v + '. Expected ' + unitmap[v][0] + '.')
    return ds


def get_constants(ds, dpath='/p/vast1/pochedls/constant/'):
    """
    get_constants(ds)

    Function gets the land fraction (sftlf) and elevation (orog) corresponding to the
    data / grid resolution (specified by the input dataset, ds).

    The function first looks to see if the constants data exists on the dataset grid and,
    if so, loads it. If the data does not exist, the method will remap (hard-coded) reference
    files to the target grid (and save the output for future use).

    Parameters:
    -----------
    ds (xr.Dataset): xr.Dataset containing fields/grid of interest

    Returns:
    --------
    xr.Dataset with constant fields (orog and sftlf) added

    Notes:
    ------
    - Original sftlf data from: https://github.com/PCMDI/pcmdi_utils/raw/main/data/navy_land.nc
    - Original orographic data from: http://research.jisao.washington.edu/data_sets/elevation/elev.0.25-deg.nc
    """
    # get grid
    lat = ds.latitude.values
    r = str(np.round(np.abs(lat[1] - lat[0]), 2))
    gridlabel = r + 'x' + r
    # check if data exists for grid
    # if so, load it and add it to the cube
    # otherwise, create constants for grid (and add them to the cube)
    fnc = dpath + '/sftlf_orog_' + gridlabel + '.nc'
    if os.path.exists(fnc):
        dsc = xr.open_dataset(fnc)
        ds['orog'] = dsc['orog']
        ds['sftlf'] = dsc['sftlf']
    else:
        import xcdat as xc
        grid = ds.drop_vars([v for v in ds.variables if v not in ['latitude', 'longitude']])
        grid = grid.bounds.add_missing_bounds()
        # get land fraction
        fn = dpath + '/navy_land.nc'
        ds_sftlf = xc.open_dataset(fn)
        ds_sftlf = ds_sftlf.regridder.horizontal('sftlf', grid, tool='xesmf', method='conservative_normed')
        ds_sftlf['sftlf'] = ds_sftlf['sftlf'].assign_attrs ({'name': 'sftlf', 'units': 'fraction'})
        ds['sftlf'] = ds_sftlf['sftlf']
        # get orography
        fn = dpath + '/elev.0.25-deg.nc'
        ds_orog = xc.open_dataset(fn)
        ds_orog = ds_orog.squeeze()
        ds_orog = ds_orog.drop_vars('time')
        ds_orog = ds_orog.regridder.horizontal('data', grid, tool='xesmf', method='conservative_normed')
        orog = ds_orog['data']
        if 'meters,' in orog.units:
            orog[:, :] = orog[:, :] / 1000.
            orog = orog.assign_attrs({'name': 'orog', 'units': 'km', 'long_name': 'elevation'})
        # mask below sea level values
        orog = orog.where(orog > 0., 0.)
        # add to dataset
        ds['orog'] = orog
        # combine datasets
        ds_sftlf['orog'] = orog
        # to netcdf
        ds_sftlf.to_netcdf(fnc)
        # clean up
        ds_sftlf.close()
        ds_orog.close()
    return ds

def get_target_vertical_grid(p, sp):
    """
    get_target_vertical_grid(p, sp)

    Function computes pressure levels that are appropriate for RTTOV. This information
    is used as a target vertical grid for 3D atmospheric data (e.g., t[pressure, lat, lon]).

    Parameters:
    -----------
    p (xr.DataArray): pressure level dataarray/column
    sp (xr.DataArray): surface pressure dataarray (lat x lon)

    Returns:
    --------
    xr.Dataset with target pressure level and pressure layer fields
    """
    # get lat / lon dataarrays for convenience
    lat = sp.latitude
    lon = sp.longitude
    # store pressure attributes (to be re-applied later)
    attrs = p.attrs
    # get pressure fields
    sp = sp.values
    p = p.values
    # get data array dimensions
    nlat = len(lat)
    nlon = len(lon)
    nlev = len(p)
    # initialize output dataarrays
    target_levels = np.zeros([nlev+1, nlat, nlon])
    target_layers = np.zeros([nlev, nlat, nlon])
    # loop over all grid cells (lat x lon loop)
    for i in range(nlat):
        for j in range(nlon):
            # choose pressure levels from top of model atmosphere (lowest pressure)
            # to the surface pressure in even spacing (nlev + 1 steps)
            plev = np.linspace(np.min(p), sp[i, j], nlev + 1)
            # take mean of plev as layer values
            play = (plev[1:] + plev[0:-1])/2.
            # add level / layer values to initialized arrays
            target_levels[:, i, j] = plev
            target_layers[:, i, j] = play
    # create dataarrays
    dalev = xr.DataArray(
        data=target_levels,
        dims=['level_index', 'latitude', 'longitude'],
        coords = {'level_index': np.arange(target_levels.shape[0]), 'latitude': lat, 'longitude': lon},
        attrs=attrs,
        name='level'
    )
    dalay = xr.DataArray(
        data=target_layers,
        dims=['layer_index', 'latitude', 'longitude'],
        coords = {'layer_index': np.arange(target_layers.shape[0]), 'latitude': lat, 'longitude': lon},
        attrs=attrs,
        name='layer'
    )
    # merge dataarrays to xr.Dataset
    ds = dalev.to_dataset()
    ds['layer'] = dalay
    # return dataset
    return ds


def interpolate_profiles(pi, p, y, extrap='linear'):
    """
    interpolate_profiles(pi, p, y)

    Function interpolates a dataarray field (y[level, lat, lon]) from input
    levels (p[level]) to a target pressure level field (pi[level, lat, lon]).

    The method either constant values for pressure levels that extend beyond the
    input grid (uses np.interp). For example, if the input grid has an upper-
    bound pressure of 1000 hPa with a temperature value of 300 K and the target 
    pressure field has a upper-bound pressure of 1003 hPa, the interpolated value
    is 300 K. Alternatively, out-of-range values can be determined via linear
    extrapolation (using scipy.interpolate.interp1d). Interpolation is performed
    in log(pressure) space.

    Parameters:
    -----------
    pi (array): target pressure field [level, lat, lon] on which to interpolate input data
    p (array): input pressure levels (p[level])
    y (array): input data values to be interpolated (y[level, lat, lon])
    extrap (str, optional): extrapolation type (linear or constant, default is linear)

    Returns:
    --------
    Numpy array of interpolated data
    """
    nlat = y.shape[1]
    nlon = y.shape[2]
    yy = np.zeros(pi.shape)*np.nan
    for i in range(nlat):
        for j in range(nlon):
            # note that we use numpy, which uses constant extrapolation
            # this was intentional
            if extrap == 'linear':
                f = scipy.interpolate.interp1d(np.log(p), y[:, i, j], fill_value='extrapolate')
                yi = f(np.log(pi[:, i, j]))
            elif extrap == 'constant':
                yi = np.interp(np.log(pi[:, i, j]), np.log(p), y[:, i, j])
            else:
                raise ValueError("Unknown extrapolation type")
            yy[:, i, j] = yi
    return yy


def adjust_3d_fields(ds, extrap='linear'):
    """
    adjust_3d_fields(ds)

    Function gets pressure levels/layers that span from the surface to the top
    of the model in equal intervals (p[level, lat, lon]). The method then re-maps
    all 3D fields to these layer values (that are consistent with RTTOV-SCATT
    requirements).

    Parameters:
    -----------
    ds (xr.Dataset): dataset containing data to be adjusted.
    extrap (str, optional): extrapolation type (linear or constant, default is linear)

    Returns:
    --------
    xr.Dataset with computed pressure levels, layers, and adjusted 3D fields.

    Notes
    -----
    The function assumes that the dataarrays correspond to a single timestep
    and that the 3d fields are organized by [level, lat, lon]. The dataset should
    include a `level` field that corresponds to the standard model pressure levels
    and a `sp` field that corresponds to the model surface pressure values.
    """
    # get existing levels
    p = ds.level.values
    # get target levels / layers
    ds3d = get_target_vertical_grid(ds.level, ds.sp)
    # get 3d vars to be interpolated
    vars_3d = [v for v in ds.variables if len(ds[v].shape) > 2]
    # loop over 3d variables and interpolate
    dims = ds3d.layer.dims
    for v in vars_3d:
        # get variable data
        data = ds[v].values
        yi = interpolate_profiles(ds3d.layer.values, p, data, extrap=extrap)
        # cast to dataarray
        da = xr.DataArray(
            data=yi,
            dims=dims,
            coords = {dims[0]: np.arange(yi.shape[0]), 'latitude': ds.latitude, 'longitude': ds.longitude},
            attrs=ds[v].attrs,
            name=v
        )
        # place data in dataset
        if ds3d is None:
            ds3d = da.to_dataset()
        else:
            ds3d[v] = da
    return ds3d


def setup_rttov_profiles(ds, ds3d, dtime, angles=[0., 0.]):
    """
    setup_rttov_profiles(ds, ds3d, dtime, angles=[0., 0.])

    Function take surface and atmospheric fields and places the data 
    into the profile structure needed to run RTTOV.

    Some fields (e.g., snow/lake cover or sea ice concentration) are used
    to determine the surface classification and emissivity coefficients (i.e.,
    they are not used directly by RTTOV). 

    Parameters:
    -----------
    ds (xr.Dataset): Dataset containing surface fields required for RTTOV (see Note). 
    ds3d (xr.Dataset): Dataset containing 3D fields required for RTTOV (see Note).
    dtime (datetime object): Datetime object with data timestamp (data is expected to represent one timestep)
    angles (list; optional) - list of zenith angle and aziumth angle (e.g., [0, 0])

    Returns:
    --------
    profiles object used as input profiles in RTTOV

    Note:
    -----
    RTTOV needs the following fields:
        Atmospheric profiles
            t - atmospheric temperature [level, lat, lon]
            q - atmospheric specific humidity [level, lat, lon]
            cc - cloud cover fraction [level, lat, lon]
            ciwc - specific cloud ice water content [level, lat, lon]
            clwc - specific cloud liquid water content [level, lat, lon]
            crwc - specific rain water content [level, lat, lon]
            cswc - specific snow water content [level, lat, lon]
            level - atmospheric pressure levels [level, lat, lon]
            layer - atmospheric pressure layers on which data is stored [level, lat, lon]
        Surface fields
            sp - surface pressure [lat, lon]
            t2m - 2m surface temperature [lat, lon]
            q2m - 2m specific humidity [lat, lon]
            u10 - 10m u-component of wind [lat, lon]
            v10 - 10m v-component of wind [lat, lon]
        Surface geometry
            latitude - grid cell latitude [lat]
            longitude - grid cell longitude [lon]
            orog - grid cell elevation [lat, lon]
        Skin fields
            skt - skin temperature [lat, lon]
            sftlf - surface land fraction [lat, lon]
            cl - lake cover fraction [lat, lon]
            snowc - snow cover [lat, lon]
            siconc - sea ice concentration [lat, lon]

    The minimum specific humidity is set to 0.1E-10 (replacing zeros).

    Emissivity Rules Applied in this Function (FASTEM Coefficients in parenthesis)
        * Surface type 0 (land) if land fraction greater than 0.5 (3. 0, 5.0, 15.0, 0.1, 0.3)
        * Surface type 0 (land) if land fraction greater than 0.5 and snow cover > 0.5 (2.9, 3.4, 27.0, 0.0, 0.0)
        * Surface type 1 (ocean/sea) if land fraction <= 0.5
        * Surface type 1 (ocean/sea) if land fraction > 0.5 and lake cover > 0.5
        * Surface type 2 (sea ice; new ice/snow) if land fraction <= 0.5 and sea ice concentration > 0.5 (2.2, 3.7, 122.0, 0.0, 0.15)
        * 35 psu for all points; 0 foam fraction
    """
    # get axes and dimensions
    nprofs = len(ds.latitude)*len(ds.longitude)
    nlays = len(ds3d.layer) 

    ## prepare 3D profiles
    t = np.reshape(ds3d.t.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    q = np.reshape(ds3d.q.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    q[q < 0] = 0.1E-10 # q cannot be zero
    layer = np.reshape(ds3d.layer.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    level = np.reshape(ds3d.level.transpose('latitude', 'longitude', 'level_index').to_numpy(), (nprofs, nlays+1), order='F')
    # deal with hydro profiles
    cc = np.reshape(ds3d.cc.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    ciwc = np.reshape(ds3d.ciwc.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    clwc = np.reshape(ds3d.clwc.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    crwc = np.reshape(ds3d.crwc.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    cswc = np.reshape(ds3d.cswc.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')
    cc = np.reshape(ds3d.cc.transpose('latitude', 'longitude', 'layer_index').to_numpy(), (nprofs, nlays), order='F')

    ## prepare datetimes
    DateTimes  = np.zeros((nprofs, 6)).astype(np.int32)
    for i in range(nprofs):
        DateTimes[i, :] = [dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, int(dtime.second)]

    # prepare angles
    # (zenangle, azangle) for each profile.
    Angles = np.tile(np.expand_dims(np.array(angles, dtype=np.float64), axis=0), (nprofs, 1)).astype(np.float64)

    ## prepare S2m
    # pressure, 2m temperature, 2m humidity, 10m u-wind, 10m v-wind
    S2m = np.zeros((nprofs, 5)).astype(np.float64)
    for i, v in enumerate(['sp', 't2m', 'q2m', 'u10', 'v10']):
        S2m[:, i] = np.reshape(np.array(ds[v]), nprofs, order='F')

    ## prepare surfgeom
    longrid, latgrid = np.meshgrid(ds.longitude, ds.latitude)
    latgrid = np.reshape(latgrid, (nprofs), order='F')
    longrid = np.reshape(longrid, (nprofs), order='F')
    oroggrid = np.reshape(ds.orog.squeeze().to_numpy(), (nprofs), order='F')
    SurfGeom = np.zeros((nprofs, 3))
    SurfGeom[:, 0] = latgrid
    SurfGeom[:, 1] = longrid
    SurfGeom[:, 2] = oroggrid

    ## Surface characteristics
    # Index 0 (surface type): 0/land, 1/sea, 2/seaice
    # Index 1 (water type): 0/fresh, 1/ocean
    SurfType = np.zeros((nprofs)).astype(np.int32)
    Skin = np.zeros((nprofs, 8)).astype(np.float64)
    # skin T, salinity, foam_frac, fastem_coefsx5
    # default Skin to actual skin temperature, 35 psu, no foam, land fastem coefs
    Skin[:, 0] = np.reshape(ds.skt.to_numpy(), (nprofs), order='F')
    Skin[:, 1] = 35.
    Skin[:, 2] = 0.
    Skin[:, 3:8] = np.tile(np.array([3.0, 5.0, 15.0, 0.1, 0.3]), (nprofs, 1)) # default for land
    # default SurfType to ocean SurfType
    SurfType[:] = 1
    # update SurfType for all land points
    sftlf = np.reshape(ds.sftlf.to_numpy(), (nprofs), order='F')
    inds = np.where((sftlf > 0.5))[0]
    SurfType[inds] = 0
    # update land points with lake cover
    cl = np.reshape(ds.cl.to_numpy(), (nprofs), order='F')
    inds = np.where((cl > 0.5) & (sftlf > 0.5))[0]
    SurfType[inds] = 1
    # update surface type for land points with snow cover (forest and snow type)
    snowc = np.reshape(ds.snowc.to_numpy(), (nprofs), order='F')
    inds = np.where((sftlf > 0.5) & (snowc > 0.5))[0]
    SurfType[inds] = 0
    Skin[inds, 3:8] = np.tile(np.array([2.9, 3.4, 27.0, 0.0, 0.0]), (len(inds), 1))
    # update surface for sea ice points (new ice with snow)
    siconc = np.reshape(ds.siconc.to_numpy(), (nprofs), order='F')
    inds = np.where(((sftlf <= 0.5) & (siconc > 0.5)))[0]
    SurfType[inds] = 2
    Skin[inds, 3:8] = np.tile(np.array([2.2, 3.7, 122.0, 0.0, 0.15]), (len(inds), 1))

    ## Assign profiles to RTTOV profiles object
    profiles = pyrttov.ProfilesScatt(nprofs, nlays)
    profiles.GasUnits = 1 # 1) kg/kg over moist air; 2) ppmv over moist air (default)
    profiles.P = layer
    profiles.T = t
    profiles.Q = q
    profiles.Ph = level
    profiles.HydroFrac = cc
    profiles.Clw = clwc
    profiles.Ciw = ciwc
    profiles.Snow = cswc
    profiles.Rain = crwc
    profiles.Angles = Angles
    profiles.S2m = S2m
    profiles.Skin = Skin
    profiles.SurfType = SurfType
    profiles.SurfGeom = SurfGeom
    profiles.DateTimes = DateTimes
    
    # return profiles object
    return profiles


def getViewAngles(instrument='MSU', altitude=840., lat=0.):
    """
      viewAngles = getViewAngles(instrument='MSU', altitude=840., lat=0.)
  
      Function returns the standard MSU view angles and the corresponding Earth 
      incidence angles. Optional arguments: 
        Argument:     default     description
        instrument    'MSU'       MSU/AMSU switch
        altitude      840.        Altitude of satellite orbit
        lat           0.          Latitude of measurement
    """    
    instrumentValues = {
      'MSU'  : np.arange(0, 47.351, 9.47),
      'AMSU' : np.arange(3.33/2., 48.34, 3.33)
    }
  
    angles = instrumentValues.get(instrument, [])
  
    # constants
    re = 6378.1370
    rp = 6356.7523
  
    lat_rad = lat * np.pi/180.
    eia = np.zeros((len(angles)))
    for i, angle in enumerate(angles):
      if angle > 0.0001:
        theta = angle * np.pi / 180.
        r = np.sqrt( ((re**2 * np.cos(lat_rad))**2 + (rp**2 * np.sin(lat_rad))**2) / ((re * np.cos(lat_rad))**2 + (rp * np.sin(lat_rad))**2) )
        b = r + altitude
        m = -1/np.tan(theta)
        x = (-b * m - np.sqrt(-b**2 + r**2 + m**2 * r**2)) / (1 + m**2)
        alpha = 90 - angle
        omega = np.arccos(x/r)*180./np.pi 
        eia[i] = 180. - alpha - omega
      else:
        eia[i] = 0.
  
    return angles, eia


def execute_amsu_rttov(ds, profiles, nprofs_per_call=22000, nthreads=3, rttov_installdir='/usr/workspace/pochedls/rttov/'):
    """
    execute_amsu_rttov(ds, profiles, nprofs_per_call=22000, nthreads=3, rttov_installdir='/usr/workspace/pochedls/rttov/')

    Function takes an input dataset (ds) and an RTTOV profiles object an executes RTTOV-SCATT for AMSU-A (NOAA-18). Returns
    the clear and all-sky brightness temperatures and the emissivities as a function of channel, satellite view angle,
    latitude, and longitude. Only data for AMSU channels 5, 7, and 9 are returned in an xarray dataset.

    The default settings are for 
        * parallel execution (3 threads, 22k profiles per call)
        * not verbose; use q2m; ApplyRegLimits=True; LuserCfrac=False; StoreRad=True
        * Gets earth incidence angles for a satellite altitude of 840 km

    Parameters:
    -----------
    ds (xr.Dataset): xarray dataset containing coordinate information (used for matrix sizing) and
                     creating output dataarrays
    profiles (pyrttov.ProfilesScatt object): RTTOV profiles object
    nprofs_per_call (int; optional): Number of profiles run per call to RTTOV
    nthreads (int; optional): Number of threads for parallel operation
    rttov_installdir (str; optional): Install directory for RTTOV

    Returns:
    --------
    xarray dataset containing RTTOV simulated brightness temperatures and emissivity
    """
    # initialize rttov
    amsuaRttov = pyrttov.RttovScatt()
    fncoef = "rtcoef_rttov13/rttov13pred54L/rtcoef_noaa_18_amsua.dat"
    fnhydro = "rtcoef_rttov13/hydrotable/hydrotable_noaa_amsua.dat"
    amsuaRttov.FileCoef = '{}/{}'.format(rttov_installdir, fncoef)
    amsuaRttov.FileHydrotable = '{}/{}'.format(rttov_installdir, fnhydro)
    amsuaRttov.Options.VerboseWrapper = False
    amsuaRttov.Options.StoreRad = True
    amsuaRttov.Options.UseQ2m = True
    amsuaRttov.Options.ApplyRegLimits = True
    amsuaRttov.Options.LuserCfrac = False
    # threading
    amsuaRttov.Options.Nthreads = nthreads
    amsuaRttov.Options.NprofsPerCall = nprofs_per_call
    # load instrument
    amsuaRttov.loadInst()
    # assign profiles
    amsuaRttov.Profiles = profiles
    # get view angles for AMSU
    view_angles, earth_incidence_angles = getViewAngles('AMSU')
    # pre-allocate matrices
    BT = np.zeros((amsuaRttov.Nchannels, len(view_angles), len(ds.latitude), len(ds.longitude))) * np.nan
    BTC = np.zeros((amsuaRttov.Nchannels, len(view_angles), len(ds.latitude), len(ds.longitude))) * np.nan
    EMIS = np.zeros((amsuaRttov.Nchannels, len(view_angles), len(ds.latitude), len(ds.longitude))) * np.nan
    for i, angle in enumerate(earth_incidence_angles):
        amsuaRttov.SurfEmis = np.ones((2, amsuaRttov.Profiles.Nprofiles, amsuaRttov.Nchannels))*-1
        profiles.Angles[:, :] = [angle, 0]
        amsuaRttov.runDirect()
        bt = np.reshape(amsuaRttov.Bt[:, :], (len(ds.latitude), len(ds.longitude), 15), order='F')
        bt = np.transpose(bt, (2, 0, 1))
        btc = np.reshape(amsuaRttov.BtClear[:, :], (len(ds.latitude), len(ds.longitude), 15), order='F')
        btc = np.transpose(btc, (2, 0, 1))
        e = np.reshape(amsuaRttov.SurfEmis[0, :, :], (len(ds.latitude), len(ds.longitude), 15), order='F')
        e = np.transpose(e, (2, 0, 1))
        BT[:, i, :, :] = bt
        BTC[:, i, :, :] = btc
        EMIS[:, i, :, :] = e
    # assign to dataarrays
    BT = xr.DataArray(name='bt',
                      data=BT,
                      dims=["channel", "view_angle", "latitude", "longitude"],
                      coords={'channel': (['channel'], np.arange(1, 16)),
                              'view_angle': (['view_angle'], view_angles),
                              'latitude': ds.latitude,
                              'longitude': ds.longitude},
                      attrs={'long_name': 'brightness_temperature',
                             'units': 'K',
                             'description': 'synthetic AMSU brightness temperature'})
    BTC = xr.DataArray(name='bt_clear',
                       data=BTC,
                       dims=["channel", "view_angle", "latitude", "longitude"],
                       coords={'channel': (['channel'], np.arange(1, 16)),
                               'view_angle': (['view_angle'], view_angles),
                               'latitude': ds.latitude,
                               'longitude': ds.longitude},
                       attrs={'long_name': 'clear_sky_brightness_temperature',
                              'units': 'K',
                              'description': 'synthetic clear-sky AMSU brightness temperature'})
    EM = xr.DataArray(name='emissivity',
                      data=EMIS,
                      dims=["channel", "view_angle", "latitude", "longitude"],
                      coords={'channel': (['channel'], np.arange(1, 16)),
                              'view_angle': (['view_angle'], view_angles),
                              'latitude': ds.latitude,
                              'longitude': ds.longitude},
                      attrs={'long_name': 'emissivity',
                             'units': '1',
                             'description': 'synthetic AMSU emissivity'})
    dsOut = BT.to_dataset()
    dsOut['bt_clear'] = BTC
    dsOut['emissivity'] = EM
    # sub-select channels
    dsOut = dsOut.sel(channel=[5, 7, 9])
    # return dataset
    return dsOut


def run_amsu_chunk(dtime,
                   grid='1x1',
                   dpath='/p/vast1/pochedls/era5/era5_hourly/',
                   nprofs_per_call=22000,
                   nthreads=3,
                   rttov_installdir='/usr/workspace/pochedls/rttov/',
                   extrap='linear'):
    """
    run_amsu_chunk(dtime,
                   grid='1x1',
                   dpath='/p/vast1/pochedls/era5/era5_hourly/',
                   nprofs_per_call=22000,
                   nthreads=3,
                   rttov_installdir='/usr/workspace/pochedls/rttov/',
                   extrap='linear')

    Parameters:
    -----------
    dtime (datetime object): datetime object corresponding to year / month of interest
    grid (str, optional): grid resolution (e.g., '0.25x0.25', '0.5x0.5', '1x1')
    nprofs_per_call (int; optional): Number of profiles run per call to RTTOV
    nthreads (int; optional): Number of threads for parallel operation
    rttov_installdir (str; optional): Install directory for RTTOV
    extrap (str, optional): Extrapolation type for adjust_3d_fields (default linear)

    Returns:
    --------
    xarray dataset containing RTTOV simulated brightness temperatures and emissivity
    """
    # get data "cube" for time point and grid resolution
    ds = get_cube(dtime, grid=grid, dpath=dpath)
    # validate data cube
    ds = validate_cube(ds)
    # adjust / setup "cube" for RTTOV-SCATT
    ds3d = adjust_3d_fields(ds, extrap=extrap)
    # create profiles object
    profiles = setup_rttov_profiles(ds, ds3d, dtime, angles=[0., 0.])
    # execute RTTOV
    dsOut = execute_amsu_rttov(ds, profiles, nprofs_per_call=nprofs_per_call, nthreads=nthreads, rttov_installdir=rttov_installdir)
    # return dataset
    return dsOut
