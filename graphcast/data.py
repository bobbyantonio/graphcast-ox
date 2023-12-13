import os, sys
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd

HOME = Path(__file__).parents[1]
DATASET_FOLDER = str(HOME / 'dataset')

sys.path.append(str(HOME / 'graphcast'))

from graphcast import graphcast as gc



def format_dataarray(da):
    
    rename_dict = {'latitude': 'lat', 'longitude': 'lon' }

    da = da.rename(rename_dict)
    
    da = da.sortby('lat', ascending=True)
    da = da.sortby('lon', ascending=True)
    
    return da

def load_clean_dataarray(fp, add_batch_dim=False):
    
    da = xr.load_dataarray(fp)
    da = format_dataarray(da)
    
    if add_batch_dim:
        da = da.expand_dims({'batch': 1})

    return da

def add_datetime(ds, start: str,
                 periods: int,
                 freq: str='6h',
                 ):
    
    dt_arr = np.expand_dims(pd.date_range(start=start, periods=periods, freq=freq),0)
    ds = ds.assign_coords(datetime=(('batch', 'time'),  dt_arr))
    return ds

def convert_to_relative_time(ds, zero_time: np.datetime64):
    
    ds['time'] = ds['time'] - zero_time
    return ds

def ns_to_hr(ns_val: float):
    
    return ns_val* 1e-9 / (60*60)

def load_era5(var: str, type: str, year: int, month: int, day: int, hour: int,
              rename_to_str: str=None):
    
    da = load_clean_dataarray(os.path.join(DATASET_FOLDER, type, f'era5_{var}_{year}{month:02d}.nc'), 
                                  add_batch_dim=False)
    
    return da

def load_era5_static(year, month):
    
    static_das = {}
    rename_dict = {}

    for var in tqdm(gc.STATIC_VARS):

        if var == 'geopotential_at_surface':
            var = 'geopotential'
            
        folder_name = 'surface'
        tmp_da = load_clean_dataarray(os.path.join(DATASET_FOLDER, folder_name, f'era5_{var}_{year}{month:02d}.nc'), 
                                        add_batch_dim=False)
        tmp_da = tmp_da.isel(time=0)
        static_das[var] = tmp_da
        rename_dict[tmp_da.name] = var
    
    rename_dict['z'] = 'geopotential_at_surface'

    static_ds = xr.merge(static_das.values())
    static_ds = static_ds.rename(rename_dict)
    static_ds = static_ds.drop_vars('time')

    # Check lat values are correctly ordered
    assert static_ds.lat[0] < 0 
    assert static_ds.lat[-1] > 0
    
    return static_ds

def load_era5_surface(year, month):
    
    surf_das = {}
    rename_dict = {}

    for var in tqdm(gc.TARGET_SURFACE_VARS + gc.EXTERNAL_FORCING_VARS):

        if var == 'total_precipitation_6hr':
            var = 'total_precipitation'
            
        folder_name = 'surface'
        tmp_da = load_clean_dataarray(os.path.join(DATASET_FOLDER, folder_name, f'era5_{var}_{year}{1:02d}.nc'),
                                        add_batch_dim=True,
                                        )
        
        if var == 'toa_incident_solar_radiation':
            tmp_da = tmp_da.sel(time=pd.date_range(start=f'{year}{month:02d}01', periods=3, freq='6h'))

        surf_das[var] = tmp_da
        rename_dict[tmp_da.name] = var
        
        if var == 'total_precipitation':
            surf_das[var] = tmp_da.resample(time='6h').sum()
            rename_dict[tmp_da.name] = 'total_precipitation_6hr'
        
            
    surface_ds = xr.merge(surf_das.values())
    surface_ds = surface_ds.rename(rename_dict)

    assert sorted(surface_ds.data_vars) == sorted(gc.EXTERNAL_FORCING_VARS + gc.TARGET_SURFACE_VARS )

    # Add datetime coordinate
    surface_ds = add_datetime(surface_ds, start=f'{year}{month:02d}01', periods=3, freq='6h')
    
    return surface_ds

def load_era5_plevel(year, month):
    
    plevel_das = {}
    rename_dict = {}

    for var in tqdm(gc.TARGET_ATMOSPHERIC_VARS):

        folder_name = 'plevels'
        tmp_da = load_clean_dataarray(os.path.join(DATASET_FOLDER, folder_name, f'era5_{var}_{year}{month:02d}.nc'),
                                        add_batch_dim=True)

        plevel_das[var] = tmp_da
        rename_dict[tmp_da.name] = var

    plevel_ds = xr.merge(plevel_das.values())
    plevel_ds = plevel_ds.rename(rename_dict)

    assert sorted(plevel_ds.data_vars) == sorted(gc.TARGET_ATMOSPHERIC_VARS)
    
    # Add datetime coordinate
    plevel_ds = add_datetime(plevel_ds, start=f'{year}{month:02d}01', periods=3, freq='6h')
    
    return plevel_ds