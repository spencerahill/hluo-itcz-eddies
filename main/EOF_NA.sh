#!/bin/bash
#PBS -A UPRI0023
#PBS -q casper
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=235GB
#PBS -N EOF_NA
##PBS -m ae


python -W ignore /glade/work/hcluo/v5/EOF_NA.py

# Or run application using cray-mpich with explicit binding
# mpiexec --cpu-bind depth -n 64 -ppn 32 -d 4 ./executable_name


