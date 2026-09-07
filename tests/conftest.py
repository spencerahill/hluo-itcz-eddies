"""Fixtures shared by the test suite.

The equivalence tests import Haochang Luo's original modules directly, so that
"the package reproduces the original" is checked against the code itself
rather than against a transcription of it.  ``main/myfun.py`` and the analysis
scripts assume they are run from inside ``main/``, so the fixtures below put
that directory on ``sys.path`` for the duration of the import.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_DIR = REPO_ROOT / "main"
ADJUST_DIR = REPO_ROOT / "MSEadjust"


def _import_from(directory: pathlib.Path, module: str):
    directory = str(directory)
    inserted = directory not in sys.path
    if inserted:
        sys.path.insert(0, directory)
    try:
        return importlib.import_module(module)
    finally:
        if inserted:
            sys.path.remove(directory)


@pytest.fixture(scope="session")
def myfun():
    """Haochang Luo's ``main/myfun.py``, imported as a module."""
    return _import_from(MAIN_DIR, "myfun")


@pytest.fixture(scope="session")
def decomp_script():
    """``main/adjusted_MMC_stationary_transient_calculation.py`` as a module.

    Importing it runs only its imports and its function definitions; the
    calculation sits inside an ``if __name__ == "__main__"`` block.
    """
    return _import_from(MAIN_DIR, "adjusted_MMC_stationary_transient_calculation")


@pytest.fixture(scope="session")
def zero_mask_script():
    """``main/div_vMSE_calculation.py`` as a module.

    Used only for its ``maskout``, which is the body that fills below ground
    with 0 and which 12 of the scripts share, ``MSEadjust/main.py`` among
    them.  ``MSEadjust/main.py`` itself cannot be imported under scipy 1.17 or
    later, because its line 11 reads ``from scipy.special import sph_harm``
    and scipy removed that name; ``sph_harm`` is never called in any of the
    three files that import it.
    """
    return _import_from(MAIN_DIR, "div_vMSE_calculation")


LEVELS = np.array(
    [50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0,
     950.0, 1000.0]
)


@pytest.fixture
def grid():
    """A small (time, level, latitude, longitude) grid shaped like the archive.

    Twelve pressure levels from 50 to 1000 hPa, matching the synthetic archive
    in the manuscript repository, on 9 latitudes, 8 longitudes and 5 times.
    """
    return {
        "time": np.arange(5.0),
        "level": LEVELS,
        "latitude": np.linspace(-8.0, 8.0, 9),
        "longitude": np.linspace(0.0, 87.5, 8),
    }


@pytest.fixture
def p_sfc(grid):
    """Surface pressure in hPa, with a ridge that reaches above the 900 hPa level.

    The ridge is what makes the below-ground handling do anything.  Its minimum
    is below 900 hPa, so at least one stored level is under the surface at some
    gridpoints, and the surface falls partway through a layer at most others.
    """
    lat = xr.DataArray(grid["latitude"], dims="latitude",
                       coords={"latitude": grid["latitude"]})
    lon = xr.DataArray(grid["longitude"], dims="longitude",
                       coords={"longitude": grid["longitude"]})
    time = xr.DataArray(grid["time"], dims="time", coords={"time": grid["time"]})
    ridge = np.exp(-((lon - 40.0) / 6.0) ** 2 - ((lat - 2.0) / 3.0) ** 2)
    seasonal = 2.0 * np.sin(2 * np.pi * time / 30.0)
    return (1008.0 - 3.0 * np.cos(lat * np.pi / 12.0) + seasonal
            - 145.0 * ridge).transpose("time", "latitude", "longitude")


@pytest.fixture
def field(grid):
    """A smooth analytic 4-D field with no NaN, for quadrature checks."""
    lev = xr.DataArray(grid["level"], dims="level", coords={"level": grid["level"]})
    lat = xr.DataArray(grid["latitude"], dims="latitude",
                       coords={"latitude": grid["latitude"]})
    lon = xr.DataArray(grid["longitude"], dims="longitude",
                       coords={"longitude": grid["longitude"]})
    time = xr.DataArray(grid["time"], dims="time", coords={"time": grid["time"]})
    shape = np.sin(np.pi * lev / 1000.0)
    wave = np.sin(np.deg2rad(6 * lon) - 2 * np.pi * time / 6.0)
    return (2.5 * shape * np.cos(np.deg2rad(3 * lat)) + 1.5 * shape * wave
            ).transpose("time", "level", "latitude", "longitude")
