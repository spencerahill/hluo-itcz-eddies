"""Data roots and ERA5 file-name construction.

Every path Haochang Luo's scripts use is an absolute ``/glade`` string literal
sitting inside an ``if __name__ == "__main__"`` block, so nothing outside the
script can redirect it.  That is what forced the golden-master runner
(``code-review/synthetic/run_golden.py`` in the manuscript repository) to
rewrite the literals by text substitution.  Here the roots come from
environment variables with the GLADE locations as defaults, so the same code
runs against the real archive, against the miniature synthetic archive, or
against a copy anywhere else.

Environment variables, all optional:

``ITCZ_ERA5_ROOT``
    Root of the ERA5 collection.  Default
    ``/glade/campaign/collections/rda/data/d633000``.  Note that the directory
    is ``d633000``; the older name ``ds633.0`` stopped resolving when NCAR
    renamed RDA to GDEX in September 2025, and six files in this repository
    still carry it.
``ITCZ_GPCP_ROOT``
    Root of the GPCP daily collection.  Default
    ``/glade/campaign/collections/rda/data/d728007``.
``ITCZ_ADJUST_ROOT``
    Directory holding the ``adjustment_YYYYMM.nc`` output of the mass
    adjustment.  Default ``/glade/work/hcluo/data/MSE_adjust``.
``ITCZ_MSE_ROOT``
    Directory holding the 4-D ``MSE_*.nc`` files.  Default
    ``/glade/derecho/scratch/hcluo/MSE``.
``ITCZ_FIELDS_ROOT``
    Directory holding the monthly ``fields_YYYYMM.nc`` files that
    ``scripts/assemble_fields.py`` writes: the mass-adjusted meridional wind,
    the MSE and the surface pressure on the analysis grid, so that a yearly
    calculation reads 14 contiguous files instead of 1,700 daily ERA5 ones.
    Default ``/glade/derecho/scratch/spencerhill/itcz-fields``.
``ITCZ_PRODUCT_ROOT``
    Directory the decomposition products are written under.  Default
    ``/glade/work/spencerhill/itcz-products``.  Until 2026-09-08 the default
    was Haochang Luo's own scratch directory, which a run without ``--out``
    would have tried to write into.
"""

from __future__ import annotations

import calendar
import os
import pathlib

__all__ = [
    "PL_VARS",
    "SFC_VARS",
    "MEANFLUX_VARS",
    "VINTEG_VARS",
    "era5_root",
    "gpcp_root",
    "adjust_root",
    "mse_root",
    "fields_root",
    "product_root",
    "shift_month",
    "last_day",
    "pl_file",
    "pl_files_month",
    "sfc_file",
    "vinteg_file",
    "meanflux_files",
    "adjust_file",
    "mse_file",
    "fields_file",
    "budget_file",
    "fields_tag",
]


_DEFAULT_ERA5_ROOT = "/glade/campaign/collections/rda/data/d633000"
_DEFAULT_GPCP_ROOT = "/glade/campaign/collections/rda/data/d728007"
_DEFAULT_ADJUST_ROOT = "/glade/work/hcluo/data/MSE_adjust"
_DEFAULT_MSE_ROOT = "/glade/derecho/scratch/hcluo/MSE"
_DEFAULT_FIELDS_ROOT = "/glade/derecho/scratch/spencerhill/itcz-fields"
_DEFAULT_PRODUCT_ROOT = "/glade/work/spencerhill/itcz-products"


def _root(env_var: str, default: str) -> pathlib.Path:
    return pathlib.Path(os.environ.get(env_var, default))


def era5_root() -> pathlib.Path:
    """Root of the ERA5 collection, from ``ITCZ_ERA5_ROOT``."""
    return _root("ITCZ_ERA5_ROOT", _DEFAULT_ERA5_ROOT)


def gpcp_root() -> pathlib.Path:
    """Root of the GPCP daily collection, from ``ITCZ_GPCP_ROOT``."""
    return _root("ITCZ_GPCP_ROOT", _DEFAULT_GPCP_ROOT)


def adjust_root() -> pathlib.Path:
    """Directory of the ``adjustment_YYYYMM.nc`` files, from ``ITCZ_ADJUST_ROOT``."""
    return _root("ITCZ_ADJUST_ROOT", _DEFAULT_ADJUST_ROOT)


def mse_root() -> pathlib.Path:
    """Directory of the 4-D ``MSE_*.nc`` files, from ``ITCZ_MSE_ROOT``."""
    return _root("ITCZ_MSE_ROOT", _DEFAULT_MSE_ROOT)


def fields_root() -> pathlib.Path:
    """Directory of the monthly ``fields_YYYYMM.nc`` files, from ``ITCZ_FIELDS_ROOT``."""
    return _root("ITCZ_FIELDS_ROOT", _DEFAULT_FIELDS_ROOT)


def product_root() -> pathlib.Path:
    """Directory the flux products are written under, from ``ITCZ_PRODUCT_ROOT``."""
    return _root("ITCZ_PRODUCT_ROOT", _DEFAULT_PRODUCT_ROOT)


# ERA5 pressure-level variables: short name to (parameter code, grid label).
# The grid label distinguishes the wind components, which are stored on the
# ``uv`` spectral-to-gridpoint grid, from the scalars on the ``sc`` grid.
PL_VARS = {
    "U": ("128_131_u", "ll025uv"),
    "V": ("128_132_v", "ll025uv"),
    "T": ("128_130_t", "ll025sc"),
    "Q": ("128_133_q", "ll025sc"),
    "Z": ("128_129_z", "ll025sc"),
}

SFC_VARS = {
    "SP": ("128_134_sp", "ll025sc"),
    "TCWV": ("128_137_tcwv", "ll025sc"),
}

# ERA5 forecast mean-flux variables.  The sign convention is ERA5's own:
# positive is downward for the surface fluxes and the two top-of-atmosphere
# fluxes alike, which is why `mse.net_energy_input` combines them with the
# signs it does.
MEANFLUX_VARS = {
    "MSSHF": ("235_033_msshf", "ll025sc"),
    "MSLHF": ("235_034_mslhf", "ll025sc"),
    "MSNSWRF": ("235_037_msnswrf", "ll025sc"),
    "MSNLWRF": ("235_038_msnlwrf", "ll025sc"),
    "MTNSWRF": ("235_039_mtnswrf", "ll025sc"),
    "MTNLWRF": ("235_040_mtnlwrf", "ll025sc"),
    "MTPR": ("235_055_mtpr", "ll025sc"),
    "MER": ("235_043_mer", "ll025sc"),
}

# ERA5's hourly vertical integrals on its 137 model levels (table 162), the
# reference the pipeline recommendation of 2026-09-09 measured the
# pressure-level pipeline against.  162.62 plus 162.59 is the column energy
# whose hourly tendency is the storage term of the budget (decision D3).
VINTEG_VARS = {
    "VIPILE": ("162_062_vipile", "ll025sc"),
    "VIKE": ("162_059_vike", "ll025sc"),
    "VIMAN": ("162_066_viman", "ll025sc"),
    "VIWVN": ("162_072_viwvn", "ll025sc"),
}


def shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    """Year and month ``offset`` months away from ``(year, month)``.

    Replaces the ``before`` and ``after`` helpers, which appear 12 and 13 times
    respectively across the scripts and handle only ``offset`` of -1 and +1.
    """
    index = (year * 12 + month - 1) + offset
    return index // 12, index % 12 + 1


def last_day(year: int, month: int) -> int:
    """Number of days in the given month."""
    return calendar.monthrange(year, month)[1]


def pl_file(var: str, year: int, month: int, day: int,
            root: pathlib.Path | None = None) -> pathlib.Path:
    """Path to one daily ERA5 pressure-level file.

    ERA5 stores the pressure-level analysis one variable per day, with the
    hour range in the file name, as in
    ``e5.oper.an.pl.128_132_v.ll025uv.1997010100_1997010123.nc``.
    """
    code, grid = PL_VARS[var]
    root = era5_root() if root is None else root
    ym = f"{year}{month:02d}"
    ymd = f"{ym}{day:02d}"
    name = f"e5.oper.an.pl.{code}.{grid}.{ymd}00_{ymd}23.nc"
    return root / "e5.oper.an.pl" / ym / name


def pl_files_month(var: str, year: int, month: int,
                   root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every existing daily pressure-level file for one variable and month.

    A file is included only if it is on disk.  Haochang Luo's scripts do the
    same, by listing the month directory and testing membership with
    ``np.isin``, which lets a partial month pass silently.  Keeping that
    behavior means a caller that needs a complete month has to check the count
    itself.
    """
    root = era5_root() if root is None else root
    paths = [pl_file(var, year, month, day, root=root)
             for day in range(1, last_day(year, month) + 1)]
    return [p for p in paths if p.exists()]


def sfc_file(var: str, year: int, month: int,
             root: pathlib.Path | None = None) -> pathlib.Path:
    """Path to one monthly ERA5 single-level analysis file."""
    code, grid = SFC_VARS[var]
    root = era5_root() if root is None else root
    ym = f"{year}{month:02d}"
    name = (f"e5.oper.an.sfc.{code}.{grid}."
            f"{ym}0100_{ym}{last_day(year, month):02d}23.nc")
    return root / "e5.oper.an.sfc" / ym / name


def vinteg_file(var: str, year: int, month: int,
                root: pathlib.Path | None = None) -> pathlib.Path:
    """Path to one monthly ERA5 file of hourly model-level vertical integrals."""
    code, grid = VINTEG_VARS[var]
    root = era5_root() if root is None else root
    ym = f"{year}{month:02d}"
    name = (f"e5.oper.an.vinteg.{code}.{grid}."
            f"{ym}0100_{ym}{last_day(year, month):02d}23.nc")
    return root / "e5.oper.an.vinteg" / ym / name


def meanflux_files(var: str, year: int, month: int,
                   root: pathlib.Path | None = None,
                   pad_december: bool = True) -> list[pathlib.Path]:
    """The forecast mean-flux files covering one month, in time order.

    ERA5's forecast products are stored in half-month blocks whose boundaries
    fall at 06 UTC on the 1st and the 16th, so a calendar month is covered by
    the second block of the preceding month plus both blocks of the month
    itself.  This reproduces the file list that ``read_mean_flux`` in
    ``MSEadjust/main.py`` and ``read_data3`` in the decomposition script build,
    including their handling of the December-to-January wrap.

    ``pad_december`` reproduces one asymmetry in those two functions: for
    December alone they append a fourth block, covering 1 to 16 January of the
    following year, which no other month gets.  Both callers then subset the
    stacked result onto the times of the field being adjusted, so the extra
    block is discarded and the asymmetry changes no published number.  It is
    kept here so that this function is a pure restatement of theirs; pass
    ``pad_december=False`` for the three symmetric blocks.
    """
    code, grid = MEANFLUX_VARS[var]
    root = era5_root() if root is None else root
    stem = f"e5.oper.fc.sfc.meanflux.{code}.{grid}."
    directory = root / "e5.oper.fc.sfc.meanflux"

    def block(y0: int, m0: int, d0: int, y1: int, m1: int, d1: int) -> pathlib.Path:
        name = (f"{stem}{y0}{m0:02d}{d0:02d}06_"
                f"{y1}{m1:02d}{d1:02d}06.nc")
        return directory / f"{y0}{m0:02d}" / name

    prev_year, prev_month = shift_month(year, month, -1)
    next_year, next_month = shift_month(year, month, 1)
    files = [
        block(prev_year, prev_month, 16, year, month, 1),
        block(year, month, 1, year, month, 16),
        block(year, month, 16, next_year, next_month, 1),
    ]
    if month == 12 and pad_december:
        files.append(block(next_year, next_month, 1, next_year, next_month, 16))
    return files


def adjust_file(year: int, month: int,
                root: pathlib.Path | None = None) -> pathlib.Path:
    """Path to one month of the column mass-adjustment product."""
    root = adjust_root() if root is None else root
    return root / f"adjustment_{year}{month:02d}.nc"


def mse_file(year: int, month: int, day: int,
             root: pathlib.Path | None = None) -> pathlib.Path:
    """Path to one day of the 4-D MSE product written by ``MSE_calculation.py``."""
    root = mse_root() if root is None else root
    ymd = f"{year}{month:02d}{day:02d}"
    return root / f"MSE_{ymd}00_{ymd}18.nc"


def fields_tag(mode: str, temporal_resolution: float, interfaces: str = "logp") -> str:
    """Subdirectory of the fields root that one assembly configuration writes to.

    ``corrected_6h_logp`` is the pipeline of 2026-09-09.  The archived mode
    applies no layer thickness of its own, so its tag carries no interface
    rule.  The files of 2026-09-08, archived mode at twelve hours, sit at
    the root itself, which the empty tag names.
    """
    hours = f"{temporal_resolution:g}h"
    if mode == "archived":
        return f"archived_{hours}"
    return f"{mode}_{hours}_{interfaces}"


def fields_file(year: int, month: int,
                root: pathlib.Path | None = None, tag: str = "") -> pathlib.Path:
    """Path to one month of the assembled fields written by ``scripts/assemble_fields.py``.

    ``tag`` is the configuration subdirectory from ``fields_tag``; the empty
    default names the root itself.
    """
    root = fields_root() if root is None else root
    return root / tag / f"fields_{year}{month:02d}.nc"


def budget_file(year: int, month: int,
                root: pathlib.Path | None = None, tag: str = "") -> pathlib.Path:
    """Path to one month of the column budget diagnostics written beside the fields."""
    return fields_file(year, month, root, tag).with_name(f"budget_{year}{month:02d}.nc")
