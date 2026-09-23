##############################################################################
#### WindData & TerrainData data preparation for Devine Input             ####
#### generates an input feature netcdf to be included in Devine           #### 
#### author: Nora Helbig                                                  ####
#### last modified: 4 Feb 2026                                            ####
##############################################################################


# -*- coding: utf-8 -*-

### Module aus externen Libraries
import os
import pandas as pd
import numpy as np
import xarray as xr
import gc
from scipy.special import erf

import time

from pathlib import Path
def cd(path):
    os.chdir(os.path.expanduser(path))

#******************************************************************************************************************************************************************************

######################################################################## Definitions ##########################################################################################

### General domain specifics ###

## Grid cell sizes

## Coarse-scale input characteristics, the same as from the coarse driving meteo input data in meter
cellsize_csc = 1000

## Fine-scale input terrain characteristics (your target resolution) in meter
cellsize_fsc = 50

## Set temporal chunking for writing out
chunk_temp = 1   # e.g. 24 daily for hourly values - adapt as needed

#********************

### Folder/File names ###

## This is the path where the script lives
BASE_DIR = Path(__file__).resolve().parent

# Input folders
dirINdem   = BASE_DIR / "input_data" / "fine_dems"
dirINmeteo = BASE_DIR / "input_data" / "coarse_meteo" / "resampled"

# Output folders (file names are set below)
dirOUTdevineInput = BASE_DIR / "input_devine_features"

#### DEM filenames
## DEM's are regular grids in metric units, i.e. projected
## Make sure there is a TWO fine-scale grid cell frame around your "fn_dem_fsc_wborder", which will be removed after calculation of terrain features
## For that frame fine-scale terrain data will be NaN
## Fine-scale DEM should be geographically aligned with the coarse meteo input and and its resolution should fit evenly into the coarse resolution, e.g. 1000m/50m = 200

## Fine-scale input for subgrid terrain characteristics (can be the same as fine-scale target but without the frame there will be NaN's at the border)
fn_dem_fsc_wborder = f'dem{cellsize_fsc}m_wborder2.asc'

## Fine-scale target DEM (only necessary if fn_dem_fsc_wborder is not available)
fn_dem_fsc         = f'dem{cellsize_fsc}m.asc'

## Set to zero if you do not have the DEM with an extra border available and then also set fn_dem_fsc_wborder = fn_dem_fsc 
clipping = 1 
# fn_dem_fsc_wborder = fn_dem_fsc ## uncomment for clipping == 0

## Coarse-scale DEM - is derived here from aggregating the fine-scale DEM to the coarse-scale grid cell resolution, i.e.
## upsampled from the given fn_dem_fsc


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

######################################################################## End Definitions ######################################################################################
#******************************************************************************************************************************************************************************


######################################################################## Main program ##########################################################################################

## Free Resources Periodically
gc.collect()

#### Read DEM data ####

print(dirINdem)

## Coarse-scale DEM from subgrid mean fine-scale DEM
print("Coarse-scale DEM from upsampling fine-scale DEM.")

## Fine-scale DEM
filename = dirINdem / fn_dem_fsc
dem_fsc  = pd.read_csv(filename, sep=r"\s+", header=None, skiprows=6, na_values=["-9999"])
dem_fsc  = dem_fsc.values

## Extract header information
myheader = pd.read_csv(filename, header=[0,1,2,3,4,5], nrows=0).columns.tolist()

for element in myheader:
 print(element)
 # Extract the second part (value) from each string
 header_list = [e.split()[1] for e in element]  # split() splits on any whitespace

del element
print("The header after extracting numbers : " + str(header_list))

ncols     = pd.to_numeric(header_list[0])
nrows     = pd.to_numeric(header_list[1])

## Generate coordinate vector
xcoords = (np.arange(ncols) + 0.5)
ycoords = (np.arange(nrows) + 0.5)
del ncols, nrows

## Number of fine cells per coarse cell
numb = cellsize_csc // cellsize_fsc
print("Number of fine cells per coarse cell:", numb)

## Convert dem_fsc temporarily to xarray to use .coarsen()
dem_fsc_xr = xr.DataArray(
    dem_fsc,
    dims=["y", "x"],
    coords={
        "y": ycoords,
        "x": xcoords
    }
)

## Coarsen: Compute coarse-scale grid cells subgrid means with NaN padding at edges
xr_dem_csc = dem_fsc_xr.coarsen(y=numb, x=numb, boundary="pad").mean()

x_coarse = xcoords.reshape(-1, numb).mean(axis=1)
y_coarse = ycoords.reshape(-1, numb).mean(axis=1)
xr_dem_csc = xr_dem_csc.assign_coords(x=x_coarse, y=y_coarse)

xr_dem_csc = xr_dem_csc.to_dataset(name="dem_csc")

del dem_fsc_xr, dem_fsc, x_coarse, y_coarse,

## Fine-scale DEM
filename = dirINdem / fn_dem_fsc_wborder
dem_fsc  = pd.read_csv(filename, sep=r"\s+", header=None, skiprows=6, na_values=["-9999"])
dem_fsc  = dem_fsc.values

## Extract header information
myheader = pd.read_csv(filename, header=[0,1,2,3,4,5], nrows=0).columns.tolist()

for element in myheader:
 print(element)
 # Extract the second part (value) from each string
 header_list = [e.split()[1] for e in element]  # split() splits on any whitespace

del element
print("The header after extracting numbers : " + str(header_list))

ncols     = pd.to_numeric(header_list[0])
nrows     = pd.to_numeric(header_list[1])

## Generate coordinate vector
xcoords = (np.arange(ncols) + 0.5)
ycoords = (np.arange(nrows) + 0.5)

del ncols, nrows

#### Read resampled coarse meteo input per time step ####

## xr_meteo_fsc_data with wind direction [°] with North 0° and clockwise counting

# xr_meteo_fsc_data:
# <xarray.Dataset> 
# Dimensions:       (time: , y: , x: )
# Coordinates:
#   * time          (time) datetime64[ns] 
#   * x             (x) float64 
#   * y             (y) float64 
# Data variables:
#     wd   (time, y, x) float64 ...

print(dirINmeteo)

filename = dirINmeteo / fn_cscmeteo

xr_meteo_fsc_data = xr.open_dataset(filename, engine="netcdf4")
print(xr_meteo_fsc_data.dims)

# Drop variables and rename (if necessary) - we only need resampled coarse-scale wind direction here

print(xr_meteo_fsc_data)

## Rename wind direction to "wd_csc_fsc" [in degrees]
xr_meteo_fsc_data = xr_meteo_fsc_data.rename({"wd": "wd_csc_fsc"})

## Number of time steps in meteo data:
time_steps = len(xr_meteo_fsc_data.time)
print("Number of time steps in data set:", time_steps)

## Extract these dates from the coarse-scale input. Gaps will be filled with NaN
start_time = xr_meteo_fsc_data.time.isel(time=0).item()
end_time   = xr_meteo_fsc_data.time.isel(time=-1).item()

print("Start - End time of meteo data set:", start_time, end_time)

## Number of time steps in meteo data:
# Full expected time axis
full_time = pd.date_range(
    start = start_time,
    end   = end_time,
    freq="1h"
)
print("Number of time steps in meteo data:", len(full_time))

## Free Resources Periodically
gc.collect()

#********************

#### Derive fine-scale Terrain features for Devine ####

## Set coords to numbers instead of coordinates
nrows, ncols = dem_fsc.shape

## Calculate local dhdxdyquad (mean square slope) [e.g. Helbig et al., 2017]

## Initialize matrices with NaN values

## Elevation derivates
dheastdx  = np.full((nrows, ncols), np.nan)
dhnorthdy = np.full((nrows, ncols), np.nan)

## Normal components
sx = np.full((nrows, ncols), np.nan)
sy = np.full((nrows, ncols), np.nan)

## Maximum indices
maxnrows = nrows - 1
maxncols = ncols - 1

## Loop through each cell (skipping borders)
for i in range(1, maxnrows):
    for j in range(1, maxncols):
        ## Calculate gradients
        dhnorthdy[i, j] = (dem_fsc[i - 1, j] - dem_fsc[i, j]) / cellsize_fsc
        dheastdx[i, j]  = (dem_fsc[i, j + 1] - dem_fsc[i, j]) / cellsize_fsc
        
        ## Calculate sx and sy according to Corripio (2002)
        sx[i, j] = 0.5 * cellsize_fsc * (dem_fsc[i, j] - dem_fsc[i, j + 1] + dem_fsc[i - 1, j] - dem_fsc[i - 1, j + 1])
        sy[i, j] = 0.5 * cellsize_fsc * (dem_fsc[i, j] + dem_fsc[i, j + 1] - dem_fsc[i - 1, j] - dem_fsc[i - 1, j + 1])


## Normalize sx and sy
sz = cellsize_fsc**2
sx = sx / sz
sy = sy / sz
sz = 1

## Compute dhdxdyquad [Sec. 2.1 from Helbig et al., 2024]
dhdxdyquad_fsc = dheastdx**2 + dhnorthdy**2

## Compute terrain azimuth angles counting from North Clockwise [similar to Hodgson, 1998]

## Compute slope [according to Corripio, 2003]
slopes_fsc = np.arccos(sz / np.sqrt(sx**2 + sy**2 + sz**2))

## Initialize azi_fsc matrix with NaN
azi_fsc = np.full((nrows, ncols), np.nan)

## South-facing
azi_fsc[(slopes_fsc > 0) & (sx == 0) & (sy < 0)] = np.pi
## North-facing
azi_fsc[(slopes_fsc > 0) & (sx == 0) & (sy >= 0)] = 0

## Slopes with sx > 0
ind = np.where((slopes_fsc > 0) & (sx > 0))
azi_fsc[ind] = (np.pi / 2 + 
              np.arctan(-(sy[ind] / np.sqrt(sx[ind]**2 + sy[ind]**2 + sz**2)) /
                         (sx[ind] / np.sqrt(sx[ind]**2 + sy[ind]**2 + sz**2))))

## Slopes with sx < 0
ind = np.where((slopes_fsc > 0) & (sx < 0))
azi_fsc[ind] = (1.5 * np.pi + 
              np.arctan(-(sy[ind] / np.sqrt(sx[ind]**2 + sy[ind]**2 + sz**2)) /
                         (sx[ind] / np.sqrt(sx[ind]**2 + sy[ind]**2 + sz**2))))

del sx, sy, sz

## Terrain with slopes_fsc of zero gets south-facing azimuth
azi_fsc[np.round(slopes_fsc * 180 / np.pi, 0) == 0] = np.pi

## Convert to degrees
azi_fsc    = azi_fsc    * 180 / np.pi
slopes_fsc = slopes_fsc * 180 / np.pi

## Print range of computed azimuth angles
print("Range of computed azimuth angles:")
print(np.nanmin(azi_fsc), np.nanmax(azi_fsc))

## Print range of computed slope angles
print("Range of computed slope angles:")
print(np.nanmin(slopes_fsc), np.nanmax(slopes_fsc))

# Clean up
del slopes_fsc

## Create an empty fine-scale terrain dataset
xr_terrain_fsc_data = xr.Dataset(
    {
        "dem_fsc": (("y", "x"), dem_fsc),
        "azi_fsc": (("y", "x"), azi_fsc),
        "dhdxdyquad_fsc": (("y", "x"), dhdxdyquad_fsc)    
    },
    coords={
        "x": xcoords,
        "y": ycoords
    }
)

del xcoords, ycoords, ncols, nrows
del dheastdx, dhnorthdy 
del azi_fsc, dem_fsc, dhdxdyquad_fsc

## Now clip two rows at north and south border and two cols at east and west border if clipping == 1 (if DEM with extra two fine-scale grid cell border frame is available, see Definitions above):
if clipping == 1:
    xr_terrain_fsc_data = xr_terrain_fsc_data.isel(
        y=slice(2, -2),   # skip 2 at top and bottom
        x=slice(2, -2)    # skip 2 at left and right
    )

## Renumber coordinates to start from 0
xr_terrain_fsc_data = xr_terrain_fsc_data.assign_coords(
    y = ("y", np.arange(xr_terrain_fsc_data.sizes['y']) + 0.5),
    x = ("x", np.arange(xr_terrain_fsc_data.sizes['x']) + 0.5),
)

## Free Resources Periodically
gc.collect()

### Derive fine-scale RelDEM terrain feature, which requires subgrid mean DEM from each coarse-scale grid cell within the domain 

## Resample coarse-scale subgrid mean "dem_csc" to fine-scale 

## Expand coarse-scale subgrid mean dem back to fine grid (no interpolation here)
xr_dem_csc_expanded = xr_dem_csc['dem_csc'].reindex(x=xr_terrain_fsc_data.x, y=xr_terrain_fsc_data.y, method="nearest")

## Interpolate coarse-scale data to fine-scale grid
xr_dem_csc_resampled = xr_dem_csc_expanded.rolling(x=numb, y=numb, center=True, min_periods=1).mean() ## rolling mean
del xr_dem_csc_expanded

xr_dem_csc_resampled = xr_dem_csc_resampled.to_dataset(name="dem_csc")

## Compute RelDEM
xr_terrain_fsc_data['reldem_fsc'] = (xr_terrain_fsc_data['dem_fsc'] - xr_dem_csc_resampled['dem_csc'])

del xr_dem_csc, xr_dem_csc_resampled

## Free Resources Periodically
gc.collect()

#### Convert the center coordinates back to the orginal for writing out

## Assign coordinates to the existing dimensions
xr_terrain_fsc_data = xr_terrain_fsc_data.assign_coords(
    x = (xr_terrain_fsc_data.x - 0.5).astype(int),
    y = (xr_terrain_fsc_data.y - 0.5).astype(int)
)

## For meteo data set to numbers instead of coordinates (true coordinates not needed in Devine)
xr_meteo_fsc_data = xr_meteo_fsc_data.assign_coords(
    x=("x", np.arange(xr_meteo_fsc_data.sizes["x"])),
    y=("y", np.arange(xr_meteo_fsc_data.sizes["y"])),
)

#********************

#### Derive fine-scale terrain features that rely on coarse wind direction ####

### Chunk dataset before computing temporal fine-scale variables
##  Adapt this as you need it:
xr_meteo_fsc_data   = xr_meteo_fsc_data.chunk({"time": chunk_temp})                 ## weekly chunk for hourly data, e.g. 168
xr_terrain_fsc_data = xr_terrain_fsc_data.chunk({"y": 200, "x": 200})

#### Compute Delta psi [0°..+/-90°] using coarse wind direction and terrain azimuth

# Derive deviation between coarse wind direction wd_csc and terrain azimuth azi
# by rotating azi to wd_csc, i.e. origin to wd_csc and counting counterclockwise:
dpsi_fsc = xr_terrain_fsc_data['azi_fsc']

# Identify the positions where azi exceeds the threshold
mask = xr_terrain_fsc_data['azi_fsc'] <= xr_meteo_fsc_data['wd_csc_fsc']

# Set values at positions where azi exceeds the threshold
dpsi_fsc = dpsi_fsc.where(~mask, (xr_meteo_fsc_data['wd_csc_fsc'] - dpsi_fsc))

# Identify the positions where azi exceeds the threshold
mask = xr_terrain_fsc_data['azi_fsc'] > xr_meteo_fsc_data['wd_csc_fsc']
# Set values at positions where azi exceeds the threshold
dpsi_fsc = dpsi_fsc.where(~mask, (xr_meteo_fsc_data['wd_csc_fsc'] - dpsi_fsc + 360))

# Sort dpsi according to +/- up until 180°:
# Identify the positions where dpsi exceeds the threshold
mask = dpsi_fsc >= 180
# Set values at positions where dpsi exceeds the threshold
dpsi_fsc = dpsi_fsc.where(~mask, (360 - dpsi_fsc))

dpsi_fsc = 90 - dpsi_fsc

dpsi_fsc = dpsi_fsc * np.pi/180

xr_terrain_fsc_data = xr_terrain_fsc_data.assign(dpsi_fsc = dpsi_fsc)
del dpsi_fsc

#### Compute Ydsc for downscaling vertical wind speed (to fine-scale) [from Helbig et al., 2024]

# Ydsc [Eq. 6 in Helbig et al, 2024]
erf_fct  = erf(0.6298 * xr_terrain_fsc_data['dpsi_fsc'])
ydsc_fsc = (-0.087122 - 0.4788 * xr_terrain_fsc_data['dpsi_fsc'] + 2.068 * erf_fct) * (-0.046577 + np.sqrt(xr_terrain_fsc_data['dhdxdyquad_fsc']/2)**0.72451)
del erf_fct

xr_terrain_fsc_data = xr_terrain_fsc_data.assign(ydsc_fsc = ydsc_fsc)

del ydsc_fsc

#### Compute diverting factor (== deflection) according to Ryan [1977] and Liston and Elder [2006] but slightly adapted here
sigma_s       = (np.arctan(np.sqrt(xr_terrain_fsc_data['dhdxdyquad_fsc']))) * np.cos((xr_meteo_fsc_data['wd_csc_fsc'] - xr_terrain_fsc_data['azi_fsc']) * np.pi/180)
divMatrix     = - 0.5 * (sigma_s) * np.sign(sigma_s) * np.sin(2 * (xr_terrain_fsc_data['azi_fsc'] - xr_meteo_fsc_data['wd_csc_fsc']) * np.pi/180) ## includes an adaptation
del sigma_s

## Transposing here is only for getting the correct: time, y, x ordering 
xr_terrain_fsc_data["deflLis_fsc"] = divMatrix.transpose("time", "y", "x")

del divMatrix

## Free Resources Periodically
gc.collect()

#********************

#### Rename for Devine convention ###

xr_terrain_fsc_data = xr_terrain_fsc_data.rename({'ydsc_fsc': 'ydsc'})
xr_terrain_fsc_data = xr_terrain_fsc_data.rename({'deflLis_fsc': 'deflLis'})
xr_terrain_fsc_data = xr_terrain_fsc_data.rename({'reldem_fsc': 'reldem'})

#********************
## Free Resources Periodically
gc.collect()
#********************

#### Write out for Devine ####

## Assign time to full time, gaps will be filled with NaN
xr_terrain_fsc_data = xr_terrain_fsc_data.reindex(time=full_time)

## True time length
time_steps = xr_terrain_fsc_data.sizes["time"]

## Free Resources Periodically
gc.collect()

print("Start writing out:")

# Time reading
tic = time.perf_counter()

# Variables to export
var_names = ['ydsc', 'deflLis', 'reldem']

## Write out as netcdf as direct input to Devine (no more preprocessing)
## netcdf-format: ##
## data variables (sorted alphabetically!!)
## deflLis (n, nrows, ncols) float64 ...
## reldem  (n, nrows, ncols) float64 ...
## ydsc    (n, nrows, ncols) float64 ...
## Dimensions: n: X nrows: x ncols: y
## Coordinates: (name comes out as defined here, same for dtype; up to 4 coordinates; below are two examples) 
## t (n) int64 ...
## time (n) datetime64[ns] ...

## Prepare output folder

print(dirOUTdevineInput)
print(os.getcwd()) 

filename = dirOUTdevineInput / "devine_input.nc"

if filename.exists():
   filename.unlink()   # removes the file

# Drop specific variables
xr_terrain_fsc_data = xr_terrain_fsc_data.drop_vars(["dem_fsc","azi_fsc", "dpsi_fsc", "dhdxdyquad_fsc"])  

# Devine input features in ALPHABETICAL order (order is necessary for Devine)
var_name = ['deflLis', 'reldem', 'ydsc']

xr_terrain_fsc_data = xr_terrain_fsc_data[var_name]

# Ensure all data variables share the same dimension order
xr_terrain_fsc_data = xr_terrain_fsc_data.transpose("time", "y", "x")

# Replace time coordinate with integer index n = 0..N-1 
xr_terrain_fsc_data = xr_terrain_fsc_data.assign_coords(n=("time", np.arange(xr_terrain_fsc_data.sizes["time"])))

# Swap dimension: time to n (time remains a coordinate on dimension n)
xr_terrain_fsc_data = xr_terrain_fsc_data.swap_dims({"time": "n"})

# Rename spatial dims
xr_terrain_fsc_data = xr_terrain_fsc_data.rename({"y": "nrows", "x": "ncols"})

# Make reldem time-dependent by repeating it for each n and transpose n, nrows, ncols
# Get the chunking of one of the existing time-dependent variables and chunk the same way
xr_terrain_fsc_data['reldem'] = (xr_terrain_fsc_data['reldem'].expand_dims(n=xr_terrain_fsc_data.sizes['n'])
                                    .transpose('n', 'nrows', 'ncols')
                                    .chunk(xr_terrain_fsc_data['deflLis'].chunksizes)
)

# Set the same chunking for all variables (time-dependent) - adapt for your domain size and time resolution
encoding = {var: {'zlib': False, 'chunksizes': (chunk_temp, 200, 200)} for var in xr_terrain_fsc_data.data_vars}

# Save to NetCDF - adapt chunking as necessary
xr_terrain_fsc_data.chunk({'n': chunk_temp, 'nrows': 200, 'ncols': 200}).to_netcdf(
    filename,
    engine="netcdf4",
    encoding=encoding
)

print("Done writing for Devine input")

toc = time.perf_counter()
print(f"Finished writing out in {toc - tic:0.4f} seconds")

print('Devine input features as written to netcdf file:')
print(xr_terrain_fsc_data)

## Free Resources Periodically
gc.collect()

#********************

