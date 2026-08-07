#!/bin/bash -l
#PBS -N main_v
#PBS -A UPRI0023
### Each array subjob will be assigned a single CPU with 4 GB of memory
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=235GB
#PBS -l walltime=12:00:00
#PBS -q main
### Request 10 subjobs with array indices spanning 1980-2023 (input year)
#PBS -J 1980-2022:2
#PBS -j oe
#PBS -m ae

##export TMPDIR=${SCRATCH}/temp
##mkdir -p ${TMPDIR}

### Run program

python -W ignore /glade/work/hcluo/pro1/main_v.py $PBS_ARRAY_INDEX