#!/bin/bash -l
#PBS -N climatology_corr_calculate
#PBS -A UPRI0023
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=360GB
#PBS -l walltime=24:00:00
#PBS -q casper

#PBS -m ae

##export TMPDIR=${SCRATCH}/temp
##mkdir -p ${TMPDIR}

### Run program

python -W ignore /glade/work/hcluo/v5/climatology_corr_calculate.py