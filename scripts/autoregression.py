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
from graphcast import graphcast as gc
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree

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
sys.path.append('/home/a/antonio/repos/graphcast-ox/graphcast/data')

import data

DATASET_FOLDER = '/home/a/antonio/repos/graphcast-ox/dataset'
OUTPUT_VARS = [
    '2m_temperature', 'total_precipitation_6hr', '10m_v_component_of_wind', '10m_u_component_of_wind', 'specific_humidity'
]

params_file = SimpleNamespace(value='/home/a/antonio/repos/graphcast-ox/params/params_GraphCast-ERA5_1979-2017-resolution_0.25-pressure_levels_37-mesh_2to6-precipitation_input_and_output.npz')

    # @title Load the model
with open(f"{params_file.value}", 'rb') as f:
    ckpt = checkpoint.load(f, gc.CheckPoint)
params = ckpt.params

model_config = ckpt.model_config
task_config = ckpt.task_config

with open("/home/a/antonio/repos/graphcast-ox/stats/diffs_stddev_by_level.nc","rb") as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open("/home/a/antonio/repos/graphcast-ox/stats/mean_by_level.nc", "rb") as f:
    mean_by_level = xr.load_dataset(f).compute()
with open("/home/a/antonio/repos/graphcast-ox/stats/stddev_by_level.nc","rb") as f:
    stddev_by_level = xr.load_dataset(f).compute()

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
    model_config: gc.ModelConfig,
    task_config: gc.TaskConfig):
    """Constructs and wraps the GraphCast Predictor."""
    
    # Deeper one-step predictor.
    predictor = gc.GraphCast(model_config, task_config)

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
    parser.add_argument('--load-era5', action='store_true',
                        help='Load from ERA5') 
    args = parser.parse_args()
    year = args.year
    month = args.month
    
    logger.info(f'Platform: {xla_bridge.get_backend().platform}')

    ########
    # Static variables
    static_ds = data.load_era5_static(year=year, month=month)

    ######
    # Surface
    surface_ds = data.load_era5_surface(year=year, month=month)

    #############
    # Pressure levels 
    plevel_ds = data.load_era5_plevel(year=year, month=month)
    prepared_ds = xr.merge([static_ds, surface_ds, plevel_ds])
    prepared_ds = convert_to_relative_time(prepared_ds, prepared_ds['time'][1])

    t0 = prepared_ds['datetime'][0][1].values

    # Note: Solar radiation is assumed to be in 6hr intervals
    solar_radiation_ds = load_clean_dataarray(os.path.join(DATASET_FOLDER, f'surface/era5_toa_incident_solar_radiation_{year}{month:02d}.nc'), add_batch_dim=True)
    solar_radiation_ds = add_datetime(solar_radiation_ds, start=solar_radiation_ds['time'].values[0], periods=len(solar_radiation_ds['time']), freq='6h')
    solar_radiation_ds = convert_to_relative_time(solar_radiation_ds, zero_time=t0)

    ds_slice = prepared_ds.isel(time=slice(-1, None))
    ds_final_datetime = ds_slice['datetime'][0][0]
    ds_final_time = ds_slice['time'][0]

    dts_to_fill = np.array([dt.values for dt in solar_radiation_ds['datetime'][0] if dt > ds_final_datetime])
    ts_to_fill = dts_to_fill - t0
    future_forcings = solar_radiation_ds.sel(time=ts_to_fill).to_dataset()
    future_forcings = future_forcings.rename({'tisr': 'toa_incident_solar_radiation'})

    task_config_dict = dataclasses.asdict(task_config)

    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                prepared_ds, target_lead_times=slice("6h", "6h"),
                **task_config_dict)
    
    predictor_fn = run_forward_jitted
    rng=jax.random.PRNGKey(0)
    targets_template=targets * np.nan
    verbose=True
    num_steps_per_chunk = 1
    ############################

    inputs = xr.Dataset(inputs)
    targets_template = xr.Dataset(targets_template)
    forcings = xr.Dataset(forcings)

    if "datetime" in inputs.coords:
        del inputs.coords["datetime"]

    if "datetime" in targets_template.coords:
        output_datetime = targets_template.coords["datetime"]
        del targets_template.coords["datetime"]
    else:
        output_datetime = None

    if "datetime" in forcings.coords:
        del forcings.coords["datetime"]

    num_target_steps = targets_template.dims["time"]
    num_chunks, remainder = divmod(num_target_steps, num_steps_per_chunk)
    if remainder != 0:
        raise ValueError(
            f"The number of steps per chunk {num_steps_per_chunk} must "
            f"evenly divide the number of target steps {num_target_steps} ")

    if len(np.unique(np.diff(targets_template.coords["time"].data))) > 1:
        raise ValueError("The targets time coordinates must be evenly spaced")

    # Our template targets will always have a time axis corresponding for the
    # timedeltas for the first chunk.
    targets_chunk_time = targets_template.time.isel(
        time=slice(0, num_steps_per_chunk))

    current_inputs = inputs
    # Target template is fixed
    current_targets_template = targets_template.isel(time=slice(0,1))

    predictions = []
    for chunk_index in tqdm(range(args.num_steps)):
        
        if chunk_index == 0:
            current_forcings = forcings
        else:
            current_forcings = future_forcings.isel(time=slice(chunk_index-1, chunk_index))
            data_utils.add_derived_vars(current_forcings)
            current_forcings = current_forcings[list(task_config_dict['forcing_variables'])].drop_vars("datetime")

        actual_target_time = current_forcings.coords["time"]  
        current_forcings = current_forcings.assign_coords(time=targets_chunk_time)
        current_forcings = current_forcings.compute()
        
        # Make predictions for the chunk.
        rng, this_rng = jax.random.split(rng)
        prediction = predictor_fn(
            rng=this_rng,
            inputs=current_inputs,
            targets_template=current_targets_template,
            forcings=current_forcings)

        next_frame = xr.merge([prediction, current_forcings])

        current_inputs = rollout._get_next_inputs(current_inputs, next_frame)
        prediction = prediction.assign_coords(time=actual_target_time)
        predictions.append(prediction)
        
        prediction[OUTPUT_VARS].isel(level=len(gc.PRESSURE_LEVELS_ERA5_37)-1).to_netcdf(os.path.join(args.output_dir, f'pred_{year}{month:02d}01_n{chunk_index}.nc'))

    logger.info('Complete')
