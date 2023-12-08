#!/bin/bash 
#SBATCH --job-name=test
#SBATCH --output=era5-%A.txt 
#SBATCH --partition=shared 
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00 
#SBATCH --mem-per-cpu=100gb

source ~/.bashrc
conda activate base

# srun python /home/a/antonio/repos/graphcast-ox/data_prep/fetch_era5.py --year 2016 --output-dir /home/a/antonio/repos/graphcast-ox/dataset --surface --months 1 --days 1;
# srun python /home/a/antonio/repos/graphcast-ox/data_prep/fetch_era5.py --year 2016 --output-dir /home/a/antonio/repos/graphcast-ox/dataset --plevels --months 1 --days 1;
srun python /home/a/antonio/repos/graphcast-ox/data_prep/fetch_era5.py --year 2016 --output-dir /home/a/antonio/repos/graphcast-ox/dataset --vars toa_incident_solar_radiation --months 1 2;