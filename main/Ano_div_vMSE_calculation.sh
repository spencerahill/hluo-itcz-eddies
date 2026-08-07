#!/bin/bash -l
#PBS -N Ano_div_vMSE_calculation
#PBS -A UCCN0006
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=235GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -J 1997-2023:1
#PBS -j oe
#PBS -m ae

##export TMPDIR=${SCRATCH}/temp
##mkdir -p ${TMPDIR}

### Run program

python -W ignore /glade/work/hcluo/v5/Ano_div_vMSE_calculation.py $PBS_ARRAY_INDEX