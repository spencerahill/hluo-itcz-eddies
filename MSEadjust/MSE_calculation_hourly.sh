#!/bin/bash
#PBS -A UPRI0023
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -N MSE_calculation_hourly
#PBS -m ae

# Load modules to match compile-time environment
module --force purge
module load ncarenv/23.09 intel-oneapi/2023.2.1 craype/2.7.23 cray-mpich/8.1.27

# Run application with MPI binding helper script
mpibind python -W ignore /glade/work/hcluo/pro1/MSE_calculation_hourly.py

# Or run application using cray-mpich with explicit binding
# mpiexec --cpu-bind depth -n 64 -ppn 32 -d 4 ./executable_name



