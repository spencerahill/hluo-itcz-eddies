# ITCZ transient-eddy MSE flux

Analysis code for a manuscript on the meridional moist static energy (MSE)
flux by transient eddies near the ITCZ in ERA5, by Haochang Luo, Spencer Hill
and Michela Biasutti.

## Two code trees

`main/` and `MSEadjust/` hold the original scripts, as Haochang Luo wrote and
ran them on NCAR's Derecho and Casper. They are the specification of what the
published figures show, and they are what the golden-master test runs against.

One line has been changed in them. `MSEadjust/main.py`, `main_v.py` and
`examine.py` each carried `from scipy.special import sph_harm` and never called
it, and scipy removed that name, so all three failed at import before executing
a statement. The import is deleted; nothing else in either directory differs
from what Haochang Luo ran.

`itcz_eddies/` is a Python package holding the same calculations as importable,
testable functions, with the duplication removed and with the pieces that
`puffins` already provides taken from there. `scripts/` holds thin entry
points that call it, `jobs/` a single parameterized PBS template, and `tests/`
the test suite.

The two trees are kept side by side on purpose. Every function in
`itcz_eddies/` that reproduces original behavior is tested against the
original code, so a difference in any published number is a deliberate change
with a test that names it.

## Installing

```
conda env create -f environment.yml
conda activate itcz-eddies
pip install -e . --no-deps
pytest
```

`puffins` (https://github.com/spencerahill/puffins) is a dependency and is
installed the same way from its own checkout. `windspharm` needs a Fortran
compiler and comes from conda-forge rather than pip.

## Running a year

Two entry points. `scripts/assemble_fields.py` reads one month of ERA5 and
the archived mass adjustment and writes the adjusted wind, the MSE and the
surface pressure to one file under `ITCZ_FIELDS_ROOT`; the fourteen months a
year needs (the year plus a flank on each side for the time mean) run as
independent jobs. `scripts/decomposition.py` then reads those files, or the
daily ERA5 files directly when they are absent, and writes the
column-integrated flux terms under `ITCZ_PRODUCT_ROOT` in a directory named
for the configuration. `jobs/submit.sh` is the PBS template for both:

```
qsub -v "SCRIPT=assemble_fields.py,ARGS=--year 1997 --month 7,CONDA_ENV=<prefix>" jobs/submit.sh
qsub -v "SCRIPT=decomposition.py,YEAR=1997,ARGS=--quadrature mass --time-mean lanczos --zonal-mean mass-weighted,CONDA_ENV=<prefix>" jobs/submit.sh
```

## Where the data lives

The original scripts carry absolute `/glade` paths as string literals inside
their `__main__` blocks. `itcz_eddies.paths` reads the same locations from
environment variables instead, with the GLADE paths as defaults, so the
package runs against the real archive, against the miniature synthetic archive
used by the tests, or against a copy anywhere else. `itcz_eddies/paths.py`
lists the variables.
