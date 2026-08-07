#!/bin/bash
#PBS -A UPRI0023
#PBS -q casper
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=235GB
#PBS -N line_unadj
##PBS -m ae


python -W ignore /glade/work/hcluo/v5/line_unadj.py

# Or run application using cray-mpich with explicit binding
# mpiexec --cpu-bind depth -n 64 -ppn 32 -d 4 ./executable_name


