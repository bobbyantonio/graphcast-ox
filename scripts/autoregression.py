'''
Script for rolling out graphcast predictions

On Oxford's A100, it currently runs out of memory when trying to load in enough data for between 80-100 time steps (80 works, 100 doesn't)

'''

import dataclasses
import functools
from types import SimpleNamespace
from argparse import ArgumentParser

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from graphcast import data

import os, sys
import datetime
import haiku as hk
import jax
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas as pd
import logging
from jax.lib import xla_bridge

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

sys.path.append('/home/a/antonio/repos/graphcast-ox/data_prep')
sys.path.append('/home/a/antonio/repos/graphcast-ox/graphcast')

from graphcast import graphcast as gc

DATASET_FOLDER = '/home/a/antonio/repos/graphcast-ox/dataset'
OUTPUT_VARS = [
    '2m_temperature', 'total_precipitation_6hr', '10m_v_component_of_wind', '10m_u_component_of_wind', 'specific_humidity'
]

params_file = SimpleNamespace(value='/home/a/antonio/repos/graphcast-ox/params/params_GraphCast-ERA5_1979-2017-resolution_0.25-pressure_levels_37-mesh_2to6-precipitation_input_and_output.npz')

    # @title Load the model
with open(f"{params_file.value}", 'rb') as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params

model_config = ckpt.model_config
task_config = ckpt.task_config


def get_all_visible_methods(obj):
    return [item for item in dir(obj) if not item.startswith('_')]

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


################################
## Functions from graphcast notebook
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
    """Constructs and wraps the GraphCast Predictor."""
    
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    
    predictor = construct_wrapped_graphcast(model_config, task_config)
    
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            i, t, f)
        return loss, (diagnostics, next_state)
    
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
    return functools.partial(
        fn, model_config=model_config, task_config=task_config)

state = {}
# Always pass params and state, so the usage below are simpler
def with_params(fn):
    return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Folder to save data to")
    parser.add_argument('--num-steps', type=int, required=True,
                        help='Number of autoregressive steps to run for')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to collect data for')
    parser.add_argument('--month', type=int, default=1,
                        help='Month to start on')
    parser.add_argument('--day', type=int, default=1,
                    help='Day of month to start on')  
    args = parser.parse_args()
    year = args.year
    month = args.month
    
    logger.info(f'Platform: {xla_bridge.get_backend().platform}')
    
    prepared_data_fp = os.path.join(DATASET_FOLDER, 'prepared_inputs', f'input_{year}{month:02d}01.nc')
    

    ########
    # Static variables
    static_ds = data.load_era5_static(year=year, month=month)

    ######
    # Surface
    surface_ds = data.load_era5_surface(year=year, month=month)

    #############
    # Pressure levels 
    plevel_ds = data.load_era5_plevel(year=year, month=month)

    prepared_ds = xr.merge([surface_ds, plevel_ds])
    
    
    #TODO: I think this can be donw much more straightforwardly with reindex ?
    
    # Add forcing data for as long as needed, to be used in the rollout
    # In order to do this we need dummy target data (replace with nulls at inference time)
    # so this just uses the same values as the last available time step

    zero_time = prepared_ds['datetime'][0][1].values

    # Note: Solar radiation is assumed to be in 6hr intervals
    solar_radiation_ds = load_clean_dataarray(os.path.join(DATASET_FOLDER, f'surface/era5_toa_incident_solar_radiation_{year}{month:02d}.nc'), add_batch_dim=True)
    solar_radiation_ds = add_datetime(solar_radiation_ds, start=solar_radiation_ds['time'].values[0], periods=len(solar_radiation_ds['time']), freq='6h')
    solar_radiation_ds = convert_to_relative_time(solar_radiation_ds, zero_time=zero_time)

    ds_slice = prepared_ds.isel(time=slice(-1, None))
    ds_final_datetime = ds_slice['datetime'][0][0]
    ds_final_time = ds_slice['time'][0]

    dts_to_fill = np.array([dt.values for dt in solar_radiation_ds['datetime'][0] if dt > ds_final_datetime])
    dts_to_fill = dts_to_fill - zero_time
    future_forcings = solar_radiation_ds.sel(time=dts_to_fill)

    future_targets = []
    for i, dt in enumerate(dts_to_fill[:args.num_steps]):
        target = ds_slice.copy()
        target['time'] = target['time'] + np.timedelta64(6*(i+1), 'h')
        target['datetime'] = target['datetime'] + np.timedelta64(6*(i+1), 'h')

        external_forcing = future_forcings.isel(time=slice(i,i+1))
        target['toa_incident_solar_radiation'] = external_forcing
        
        future_targets.append(target)

    # Concat it all together

    prepared_ds = xr.concat([prepared_ds] + future_targets, dim='time')
    prepared_ds = xr.merge([static_ds,prepared_ds])
        
    with open("/home/a/antonio/repos/graphcast-ox/stats/diffs_stddev_by_level.nc","rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with open("/home/a/antonio/repos/graphcast-ox/stats/mean_by_level.nc", "rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    with open("/home/a/antonio/repos/graphcast-ox/stats/stddev_by_level.nc","rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()
        
    task_config_dict = dataclasses.asdict(task_config)
    max_lead_time = ns_to_hr(prepared_ds['time'].max())
    
    logger.info(f'Max lead time = {int(max_lead_time)}')

    ## Warm up step
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                prepared_ds, target_lead_times=slice("6h", f"{int(max_lead_time)}h"),
                **task_config_dict)
    
    prediction_ds = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings)
    
    logger.info('Writing predictions to file')
    prediction_ds[OUTPUT_VARS].isel(level=len(gc.PRESSURE_LEVELS_ERA5_37)-1).to_netcdf(os.path.join(args.output_dir, f'pred_{year}{month:02d}01_ntot{args.num_steps}.nc'))

    # solar_radiation_ds = xr.load_dataset(os.path.join(DATASET_FOLDER, 'surface/era5_toa_incident_solar_radiation_201601.nc'))
        
    # autoregressive_steps = 1
    # task_config_dict = dataclasses.asdict(task_config)
    
    # ## Warm up step
    # inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    #             prepared_ds, target_lead_times=slice("6h", "6h"),
    #             **task_config_dict)
    
    # # Keep track of target datetimes, for adding to autoregressive inputs
    # datetime_vals = prepared_ds['datetime'].values
        
    # rollout.chunked_prediction(
    #         run_forward_jitted,
    #         rng=jax.random.PRNGKey(0),
    #         inputs=inputs,
    #         targets_template=targets * np.nan,
    #         forcings=forcings)
        
    # logger.info('Generating predictions')
    # for n in tqdm(range(args.num_steps)):
        
    #     prediction_ds = rollout.chunked_prediction(
    #             run_forward_jitted,
    #             rng=jax.random.PRNGKey(0),
    #             inputs=inputs,
    #             targets_template=targets * np.nan,
    #             forcings=forcings)
        
    #     # Save a selection of the data
    #     prediction_ds[OUTPUT_VARS].isel(level=len(gc.PRESSURE_LEVELS_ERA5_37)-1).to_netcdf(os.path.join(args.output_dir, f'pred_{year}{month:02d}01_n{n}.nc'))
        
    #     # Get new inputs from predictions
    #     #TODO: there is something that is forcing this to recompile every time
    #     # Maybe down to different structure of dataset?
    #     # Need one function that loads datasets consistently
        
    #     datetime_vals = datetime_vals + np.timedelta64(6, 'h')
        
    #     prediction_ds['time'] = np.array([np.timedelta64(0, 'ns')])

    #     new_dummy_targets = targets

    #     minus_6hr_input = inputs.isel(time=1)
    #     minus_6hr_input['time'] = np.array([np.timedelta64(-21600000000000, 'ns')])
    #     minus_6hr_input = minus_6hr_input.drop_vars(gc.FORCING_VARS + gc.STATIC_VARS)
    #     new_ds = xr.concat([minus_6hr_input, prediction_ds, new_dummy_targets],dim='time')

    #     tmp_solar_radiation_ds = solar_radiation_ds.sel(time=datetime_vals[0])
    #     tmp_solar_radiation_ds['time'] = [item - tmp_solar_radiation_ds['time'].values[1] for item in tmp_solar_radiation_ds['time'].values]
    #     tmp_solar_radiation_ds = tmp_solar_radiation_ds.rename({'latitude': 'lat', 'longitude': 'lon', 'tisr': 'toa_incident_solar_radiation'})
    #     tmp_solar_radiation_ds = tmp_solar_radiation_ds.expand_dims({'batch': 1})
    #     tmp_solar_radiation_ds = tmp_solar_radiation_ds.sortby('lat', ascending=True)
    #     tmp_solar_radiation_ds = tmp_solar_radiation_ds.sortby('lon', ascending=True)
        
    #     new_ds = xr.merge([new_ds, static_ds, tmp_solar_radiation_ds])
    #     new_ds = new_ds.assign_coords(datetime=(('batch', 'time'),  datetime_vals))

    #     inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    #             new_ds, target_lead_times=slice("6h", "6h"),
    #             **task_config_dict)
