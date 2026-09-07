"""Dimension, coordinate and variable names used by the ERA5 archive.

The ERA5 files on GLADE name their dimensions ``latitude``, ``longitude``,
``level`` and ``time``.  ``puffins`` defaults to ``lat``, ``lon`` and ``plev``
(see :mod:`puffins.names`), so every call into ``puffins`` from this package
passes these strings explicitly rather than relying on the ``puffins``
defaults.

The pressure-level variable names are ERA5's own upper-case short names as they
appear in the GDEX netCDF files.
"""

LAT_STR = "latitude"
LON_STR = "longitude"
LEV_STR = "level"
TIME_STR = "time"
PHALF_STR = "phalf"

# ERA5 variable names, as stored in the GDEX files.
U_STR = "U"
V_STR = "V"
TEMP_STR = "T"
SPHUM_STR = "Q"
GEOPOT_STR = "Z"
P_SFC_STR = "SP"

# Names of the derived products this package writes.
MSE_STR = "MSE"
VMSE_COL_STR = "vMSE_col"
VMSE_COL_MMC_STR = "vMSE_col_MMC"
VMSE_COL_STAT_STR = "vMSE_col_stationary"
VMSE_COL_TRANS_STR = "vMSE_col_transient"
