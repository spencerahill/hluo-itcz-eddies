#!/bin/bash
# Submit one corrected-assembly job per month, on Casper.
#
# Each month is independent (scripts/assemble_fields.py), so a year plus the
# three flanking months either side that the decomposition's 60-day Lanczos
# half window needs is 18 jobs, submitted here in one go.
#
# Usage, from the repository on Casper:
#   bash jobs/assemble.sh 1997-07                       one month
#   bash jobs/assemble.sh --year 1997                   1996-10 through 1998-03
#   bash jobs/assemble.sh --smoke 4 --walltime 00:40:00 1997-07
#   bash jobs/assemble.sh --args "--interfaces midpoint --temporal-resolution 12" 1997-07
#
# Environment, with the defaults for Spencer's setup:
#   REPO, CONDA_ENV, ACCOUNT, LOGS, WALLTIME (03:00:00), MEM (48GB)
# The measured cost is written into the log by PBS; read the log.

set -euo pipefail

REPO=${REPO:-/glade/work/spencerhill/hluo-itcz-eddies}
CONDA_ENV=${CONDA_ENV:-/glade/work/spencerhill/conda-envs/itcz-eddies-pkg}
ACCOUNT=${ACCOUNT:-UPRI0023}
LOGS=${LOGS:-/glade/work/spencerhill/logs}
WALLTIME=${WALLTIME:-03:00:00}
MEM=${MEM:-48GB}
FIELDS_ROOT=${ITCZ_FIELDS_ROOT:-/glade/derecho/scratch/spencerhill/itcz-fields}
PRODUCT_ROOT=${ITCZ_PRODUCT_ROOT:-/glade/work/spencerhill/itcz-products}

extra=""
months=()
while [ $# -gt 0 ]; do
    case "$1" in
        --year)
            y=$2; shift 2
            for m in 10 11 12; do months+=("$((y - 1))-$m"); done
            for m in 01 02 03 04 05 06 07 08 09 10 11 12; do months+=("$y-$m"); done
            for m in 01 02 03; do months+=("$((y + 1))-$m"); done
            ;;
        --smoke) extra="$extra --smoke $2"; shift 2 ;;
        --args) extra="$extra $2"; shift 2 ;;
        --walltime) WALLTIME=$2; shift 2 ;;
        --mem) MEM=$2; shift 2 ;;
        *) months+=("$1"); shift ;;
    esac
done
if [ ${#months[@]} -eq 0 ]; then
    echo "usage: bash jobs/assemble.sh [--smoke N] [--args \"...\"] [--walltime H:MM:SS] [--mem NGB] YYYY-MM ... | --year YYYY" >&2
    exit 2
fi

mkdir -p "$LOGS"
for ym in "${months[@]}"; do
    year=${ym%-*}
    month=$((10#${ym#*-}))
    tag=$(printf '%d%02d' "$year" "$month")
    # extra arguments name the log, so a smoke and a full month never share one;
    # a test inside a command substitution would trip set -e when it fails
    suffix=""
    if [ -n "$extra" ]; then
        suffix=$(printf '%s' "$extra" | tr -d ' -')
    fi
    log="$LOGS/assemble-${tag}${suffix:+-$suffix}.log"
    rm -f "$log"
    id=$(qsub -N "assemble-$tag" \
         -v "SCRIPT=assemble_fields.py,ARGS=--year $year --month $month$extra,CONDA_ENV=$CONDA_ENV,REPO=$REPO,ITCZ_FIELDS_ROOT=$FIELDS_ROOT,ITCZ_PRODUCT_ROOT=$PRODUCT_ROOT" \
         -A "$ACCOUNT" -q casper -l "walltime=$WALLTIME" -l "select=1:ncpus=1:mem=$MEM" \
         -o "$log" "$REPO/jobs/submit.sh")
    echo "$ym $id $log"
done
