"""Analysis code for the ITCZ transient-eddy MSE flux manuscript.

The original scripts stay in ``main/`` and ``MSEadjust/`` unchanged.  This
package holds the same calculations as importable functions, with the
duplication removed and with the pieces ``puffins`` already provides taken
from there.  See ``README.md``.
"""

from . import columns, derivatives, era5, names, paths

__all__ = ["columns", "derivatives", "era5", "names", "paths"]
