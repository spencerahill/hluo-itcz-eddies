#!/bin/bash -l
# One parameterized PBS job for every entry point in scripts/.
#
# The 36 job scripts in main/ and MSEadjust/ are near-identical: all of them ask
# for select=1:ncpus=1:mpiprocs=1, none uses MPI, a GPU or compiled code, and
# both machines mount GLADE. So the queue, the account and the year range are
# the only things that vary, and they are arguments here rather than 36 files to
# edit.
#
# Why the queue matters. Derecho charges whole nodes: ncpus=1 on a 128-core node
# still costs 128 core-hours per hour. Casper charges the cores requested. The
# four jobs that ran on Derecho's main queue cost 105,984 core-hours as written
# and would have cost 828 on Casper.
#
# Usage:
#   qsub -v SCRIPT=decomposition.py \
#        -A UPRI0023 -q casper -J 1997-2023:1 \
#        -l walltime=12:00:00 -l select=1:ncpus=1:mem=64GB \
#        jobs/submit.sh
#
# Environment the job needs, set here rather than hardcoded in the Python:
#   ITCZ_ERA5_ROOT, ITCZ_ADJUST_ROOT, ITCZ_MSE_ROOT, ITCZ_PRODUCT_ROOT
# See itcz_eddies/paths.py for the defaults.

#PBS -N itcz-eddies
#PBS -j oe
#PBS -m ae

set -euo pipefail

: "${SCRIPT:?pass -v SCRIPT=<name of a file in scripts/>}"
: "${REPO:=${PBS_O_WORKDIR:-$(pwd)}}"
: "${CONDA_ENV:=itcz-eddies}"

module load conda || true
conda activate "${CONDA_ENV}"

# PBS_ARRAY_INDEX is the year for an array job, and is absent otherwise.
YEAR="${PBS_ARRAY_INDEX:-${YEAR:-}}"

echo "host          $(hostname)"
echo "date          $(date -Is)"
echo "repo          ${REPO}"
echo "script        ${SCRIPT}"
echo "year          ${YEAR:-<none>}"
echo "ERA5 root     ${ITCZ_ERA5_ROOT:-<default>}"
echo "product root  ${ITCZ_PRODUCT_ROOT:-<default>}"

cd "${REPO}"
python -W ignore "scripts/${SCRIPT}" ${YEAR:+--year "${YEAR}"}
