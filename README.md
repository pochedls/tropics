# tropics
Code to investigate TROPICS' ability to accurately measure the diurnal cycle in MSU/AMSU tropospheric temperature observations.

### Compile RTTOV

Follow rttov_install_notes to install RTTOV.

### Setup Environment

mamba create -n rttov2 -c conda-forge hdf5 gfortran_linux-64 netcdf4 netcdf-fortran ipython matplotlib xarray h5py xesmf xcdat==0.4

### Download Data

* Needed to setup copernicus for ERA5 downloads in "cds" conda environment (https://cds.climate.copernicus.eu/api-how-to)
* Downloaded the constants and orbit files
* In download/ folder (via compute that can download and via screen if needed):
        * conda activate cds
        * bash download_era5_wrapper.sh YYYY

### Run RTTOV

* In rttov folder:
    * ./submit_monthly_jobs.sh YYYY

### Process Data

* In process folder:
    * simulate_tmt.py [runs with 12 cores; use a full node]
    * sample_like_tropics.py [runs with 20 cores; use a full node; ~3 hour run time for 5 years / 6 satellites; informed by plot_satellites.py]
    * create_reference_diurnal_cycle.py
    * infer_diurnal_cycle.py [can use submit_python_job.py] (e.g., with the following configurations):
        * nadir_4sats_0.25K
        * 5fps_4sats_0.25K
