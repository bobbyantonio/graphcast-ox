# Note that this requires setup of an API key and installing cdsapi:
# Go to: https://cds.climate.copernicus.eu/api-how-to
import os, sys
import cdsapi
from tqdm import tqdm
from typing import Iterable
from argparse import ArgumentParser

sys.path.append('/home/a/antonio/repos/graphcast-ox/graphcast/')

PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)


SURFACE_VARS =  (
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'geopotential', 'land_sea_mask', 'mean_sea_level_pressure',
                'toa_incident_solar_radiation', 'total_precipitation',
)

PRESSURE_LEVEL_VARS = (
                'potential_vorticity',
                'specific_rain_water_content',
                'specific_snow_water_content',
                'geopotential',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
                'specific_humidity',
                'relative_humidity',
                'vertical_velocity',
                'vorticity',
                'divergence',
                'ozone_mass_mixing_ratio',
                'specific_cloud_liquid_water_content',
                'specific_cloud_ice_water_content',
                'fraction_of_cloud_cover')

c = cdsapi.Client()


def retrieve_data(year:int, 
                    output_fp:str,
                    var:str,
                    months:Iterable=range(1,13),
                    days:Iterable=range(1,32),
                    pressure_level=None
                    ):
    if var=='total_precipitation':
        # Collect full history for precip since it needs to be aggregated (the others are subsamples)
        time = [f'{n:02d}:00' for n in range(24)]
    else:
        time =  [f'{n:02d}:00' for n in (0,6,12,18)]

    request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var,
            'year': str(year),
            'month': months,
            'day': [f'{int(day):02d}' for day in days],
            'time': time,
        }

    
    if pressure_level is not None:
        if not isinstance(pressure_level, tuple) and not isinstance(pressure_level, list):
            pressure_level = [pressure_level]
        request['pressure_level'] = [str(lvl) for lvl in pressure_level]
    
    c.retrieve(
        'reanalysis-era5-single-levels' if pressure_level is None else 'reanalysis-era5-pressure-levels', 
        request, output_fp)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Folder to save data to")
    parser.add_argument('--year', type=int, required=True,
                        help='Year to collect data for')
    parser.add_argument('--surface', action='store_true',
                        help='Collect surface variables')
    parser.add_argument('--plevels', action='store_true',
                        help='Collect pressure level data')
    parser.add_argument('--vars', nargs='+', default=None,
                    help='Specific variable to collect data for')  
    parser.add_argument('--months', nargs='+', default=range(1,13),
                        help='Months to collect data for')
    parser.add_argument('--days', nargs='+', default=range(1,32),
                    help='Days to collect data for')  
    args = parser.parse_args()
    
    if args.vars:
        for var in args.vars:
            for month in args.months:
                padded_month =f'{int(month):02d}'
                retrieve_data(year=args.year,
                            months=[padded_month],
                            days=args.days,
                            var=var,
                            output_fp=os.path.join(args.output_dir, 'surface', f'era5_{var}_{args.year}{padded_month}.nc'))
    
    if args.surface:
        for var in SURFACE_VARS:
            for month in args.months:
                padded_month =f'{int(month):02d}'
                retrieve_data(year=args.year,
                            months=[padded_month],
                            days=args.days,
                            var=var,
                            output_fp=os.path.join(args.output_dir, 'surface', f'era5_{var}_{args.year}{padded_month}.nc'))
            
    if args.plevels:
        for var in PRESSURE_LEVEL_VARS:
            for month in args.months:
                padded_month =f'{int(month):02d}'
                retrieve_data(year=args.year,
                                    months=[padded_month],
                                    var=var,
                                    days=args.days,
                                    pressure_level=PRESSURE_LEVELS_ERA5_37,
                                    output_fp=os.path.join(args.output_dir, 'plevels', f'era5_{var}_{args.year}{padded_month}.nc'))
    
    