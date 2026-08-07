#!/bin/bash -l
#PBS -N GMS_calculate
#PBS -A UPRI0023
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=235GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -m ae
#PBS -J 2008-2023:1
#PBS -j oe

##export TMPDIR=${SCRATCH}/temp
##mkdir -p ${TMPDIR}

### Run program

python -W ignore /glade/work/hcluo/v5/GMS_calculate.py $PBS_ARRAY_INDEX