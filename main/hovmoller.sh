#!/bin/bash
#PBS -A UCCN0006
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=235GB
#PBS -N hovmoller
##PBS -m ae


python -W ignore /glade/work/hcluo/v5/hovmoller.py

# Or run application using cray-mpich with explicit binding
# mpiexec --cpu-bind depth -n 64 -ppn 32 -d 4 ./executable_name


