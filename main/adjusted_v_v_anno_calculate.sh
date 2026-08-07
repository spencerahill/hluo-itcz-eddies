#!/bin/bash -l
#PBS -N adjusted_v_v_anno_calculate
#PBS -A UPRI0023
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=235GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -J 1997-2000:1
#PBS -j oe
#PBS -m ae

##export TMPDIR=${SCRATCH}/temp
##mkdir -p ${TMPDIR}

### Run program

python -W ignore /glade/work/hcluo/v5/adjusted_v_v_anno_calculate.py $PBS_ARRAY_INDEX