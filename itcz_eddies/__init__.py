"""Analysis code for the ITCZ transient-eddy MSE flux manuscript.

The original scripts stay in ``main/`` and ``MSEadjust/`` unchanged.  This
package holds the same calculations as importable functions, with the
duplication removed and with the pieces ``puffins`` already provides taken
from there.  See ``README.md``.
"""

from . import (
    assembly,
    columns,
    decomp,
    derivatives,
    era5,
    mass,
    metrics,
    mse,
    names,
    paths,
    plotting,
    spectra,
    stats,
)

__all__ = [
    "assembly",
    "columns",
    "decomp",
    "derivatives",
    "era5",
    "mass",
    "metrics",
    "mse",
    "names",
    "paths",
    "plotting",
    "spectra",
    "stats",
]
