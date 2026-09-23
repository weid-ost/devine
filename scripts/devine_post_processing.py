######################################################################################
#### Python Postprocessing Devine output                                          ####
#### coarse wind * devine wind / 3                                                ####
#### writes postprocessed output in devine model output folder *_postprocessed.nc ####            
#### author: Nora Helbig                                                          ####
#### last modified: 8 Jan 2026                                                    ####
######################################################################################


# -*- coding: utf-8 -*-

### Module aus externen Libraries
import os
import xarray as xr
import gc
from pathlib import Path
def cd(path):
    os.chdir(os.path.expanduser(path))

# Free Resources Periodically
gc.collect()

#******************************************************************************************************************************************************************************

######################################################################## Definitions ##########################################################################################

### General domain specifics ###

## Grid cell sizes

## Fine-scale input terrain characteristics (your target resolution) in meter
cellsize_fsc = 50

## Set temporal chunking for writing out
chunk_temp = 2   # e.g. daily for hourly values - adapt as needed

#********************

### Folder/File names ###

## This is the path where the script lives
BASE_DIR = BASE_DIR = Path.cwd() #Path(__file__).resolve().parent

# Input folders
dirINdem      = BASE_DIR / "input_data" / "fine_dems"
dirINfeatures = BASE_DIR / "input_devine_features"
dirINmeteo    = BASE_DIR / "input_data" / "coarse_meteo" / "resampled" ## coarse input meteo resampled to fine_scale
dirINdevine   = BASE_DIR / "output_devine"

## Fine-scale Devine Feature Input
fn_devineFeature_fsc = 'devine_input.nc'

## Fine-scale Devine model Output
fn_devineOutput_fsc  = 'flowfield.nc'

## Fine-scale Devine model Output postprocessed, 
## which is dividing by 3 m/s (ARPS training data) and 
## multiplying by corresponding coarse wind speed (resampled to fine-scale)
fn_devineOutput_postproc_fsc  = 'flowfield_postprocessed.nc'

## Fine-scale DEM 
fn_dem_fsc = f'dem{cellsize_fsc}m.asc'

#### Meteo filename

## Your coarse-scale meteo input (xarray) resampled to fine-scale target resolution (in metric regular grid as used for DEMs)
## Variable names need to match the ones used here or need to be adapted
fn_cscmeteo = "fsc_meteo.nc"

## fsc_meteo.nc: with wind direction [°] with North 0° and clockwise counting
# <xarray.Dataset> 
# Dimensions:       (time: , y: , x: )
# Coordinates:
#   * time          (time) datetime64[ns] 
#   * x             (x) float64 
#   * y             (y) float64 
# Data variables:
#     wd   (time, y, x) float64 ...
#     wsp  (time, y, x) float64 ...


######################################################################## End Definitions ######################################################################################
#******************************************************************************************************************************************************************************


######################################################################## Main program ##########################################################################################

#### Read coarse resampled wind speed input ####

print(dirINmeteo)

filename = dirINmeteo / fn_cscmeteo

xr_meteo_fsc_data = xr.open_dataset(filename, engine="netcdf4")

## Drop wind direction - not required for post processing
xr_meteo_fsc_data = xr_meteo_fsc_data.drop_vars(["wd"])           

print(xr_meteo_fsc_data.dims)

## Free Resources Periodically
gc.collect()

#********************

#### Read Devine wind model output ####

print(dirINdevine)

filename = dirINdevine / fn_devineOutput_fsc

xr_devine_fsc_data = xr.open_dataset(filename, engine="netcdf4")

## Free Resources Periodically
gc.collect()

#********************

#### Postprocess Devine Output to real wind speeds    ####
#### Because ARPS training data set is with "3" m/s   ####
#### which needs to be corrected with the true coarse ####
#### wind speed                                       ####

## Devine Acceleration correction

for var in xr_devine_fsc_data.data_vars:
    xr_devine_fsc_data[var].data = (xr_devine_fsc_data[var].data / 3) * xr_meteo_fsc_data['wsp'].data

## Free Resources Periodically
gc.collect()

#********************

#### Write out postprocessed Devine netcdf ####

## Prepare output folder

print(dirINdevine)
# print(os.getcwd()) 

filename = dirINdevine / fn_devineOutput_postproc_fsc

print(filename)
print(filename.parent)

if filename.exists():
   filename.unlink()   # removes the file

## Set the same chunking for all variables (time-dependent) - adapt for your domain size and time resolution
encoding = {var: {'zlib': False, 'chunksizes': (chunk_temp, 200, 200)} for var in xr_devine_fsc_data.data_vars}

## Save to NetCDF - adapt chunking as necessary
xr_devine_fsc_data.chunk({'n': chunk_temp, 'nrows': 200, 'ncols': 200}).to_netcdf(
    filename,
    engine="netcdf4",
    encoding=encoding
)

print(xr_devine_fsc_data)

# Free Resources Periodically
gc.collect()

print("Done Devine model output postprocessing")

#********************