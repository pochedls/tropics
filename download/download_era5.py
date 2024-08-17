#!/usr/bin/env python

"""
This script is used for downloading ERA5 data from Copernicus

Before running this the first time, follow the steps here:
https://cds.climate.copernicus.eu/api-how-to

LLNL will require you to have a certificate in order to download from ECMWF,
so type this on your LOCAL machine:
https://www-csp.llnl.gov/content/assets/csoc/cspca.crt
and copy it over to feedback

Then run the following command:
export REQUESTS_CA_BUNDLE="./cspca.crt"

This script uses the cds environment.

# mamba create -n cds -c conda-forge cdsapi ipython

# Links I referred to:
    https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
    https://confluence.ecmwf.int/display/WEBAPI/Web-API+Troubleshooting#Web-APITroubleshooting-3.2.1.3WARNING:httplib2.URLErrorreceivedNone%3Curlopenerror[SSL:CERTIFICATE_VERIFY_FAILED]certificateverifyfailed
    https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
    https://stackoverflow.com/a/72374542
"""

import cdsapi
import sys
import os

# create cds client instance
c = cdsapi.Client()

# parameters
args = sys.argv
year = args[1]
month = args[2]

# grid = ['0.25', '0.25']  # '2/2'
# grid_label = '0.25x0.25'
grid = ['1', '1']  # '2/2'
grid_label = '1x1'
dpath = '/p/vast1/pochedls/era5/era5_hourly/'

# constants
pressure_level = ['1', '5', '10', '20', '30', '50', '70', '100',
                  '150', '200', '250', '300', '350', '400', '450',
                  '500', '550', '600', '650', '700', '750', '800',
                  '850', '925', '1000']
day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
       '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
       '31']
time = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
product_type = 'reanalysis'
file_format = 'netcdf'

variable_sets = {'t': {'product': 'reanalysis-era5-pressure-levels',
                       'variable': 'temperature'},
                 'q': {'product': 'reanalysis-era5-pressure-levels',
                       'variable': 'specific_humidity'},
                 'cc': {'product': 'reanalysis-era5-pressure-levels',
                        'variable': 'fraction_of_cloud_cover'},
                 'ciwc': {'product': 'reanalysis-era5-pressure-levels',
                          'variable': 'specific_cloud_ice_water_content'},
                 'clwc': {'product': 'reanalysis-era5-pressure-levels',
                          'variable': 'specific_cloud_liquid_water_content'},
                 'crwc': {'product': 'reanalysis-era5-pressure-levels',
                          'variable': 'specific_rain_water_content'},
                 'cswc': {'product': 'reanalysis-era5-pressure-levels',
                          'variable': 'specific_snow_water_content'},
                 'cl': {'product': 'reanalysis-era5-single-levels',
                          'variable': 'lake_cover'},
                 'sp': {'product': 'reanalysis-era5-single-levels',
                          'variable': 'surface_pressure'},
                 'skt': {'product': 'reanalysis-era5-single-levels',
                          'variable': 'skin_temperature'},
                 'tciw': {'product': 'reanalysis-era5-single-levels',
                          'variable': 'total_column_cloud_liquid_water'},
                 'tclw': {'product': 'reanalysis-era5-single-levels',
                          'variable': 'total_column_cloud_ice_water'},
                 'siconc': {'product': 'reanalysis-era5-single-levels',
                          'variable': 'sea_ice_cover'},
                 't2m': {'product': 'reanalysis-era5-single-levels',
                          'variable': '2m_temperature'},
                 'd2m': {'product': 'reanalysis-era5-single-levels',
                          'variable': '2m_dewpoint_temperature'},
                 'u10': {'product': 'reanalysis-era5-single-levels',
                          'variable': '10m_u_component_of_wind'},
                 'v10': {'product': 'reanalysis-era5-single-levels',
                          'variable': '10m_v_component_of_wind'},
                 'snowc': {'product': 'reanalysis-era5-land',
                           'variable': 'snow_cover'}}

freq = 'hourly'
print('Running download for: ' + year + '-' + month)

for v in variable_sets.keys():
    print(v)
    # define output file
    fnOut = dpath + v + '_' + freq + '_' + grid_label + '_' + str(year) + str(month).zfill(2) + '.nc'
    if os.path.exists(fnOut):
        continue
    # get variable set
    vset = variable_sets[v]
    # populate request dictionary
    requestDict = {}
    product = vset['product']
    requestDict['product_type'] = product_type
    requestDict['variable'] = vset['variable']
    requestDict['year'] = year
    requestDict['month'] = month.zfill(2)
    requestDict['day'] = day
    requestDict['time'] = time
    requestDict['format'] = file_format
    requestDict['grid'] = grid
    if 'pressure-levels' in product:
        requestDict['pressure_level'] = pressure_level
    # get data
    c.retrieve(product, requestDict, fnOut)
