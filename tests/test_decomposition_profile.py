"""``scripts/decomposition.py --mass-correction-profile`` on miniature fields files.

The files mimic the corrected assembly's ``fields_YYYYMM.nc`` for 1997 and
its three flanking months either side, on two latitudes, four longitudes and
four levels at six-hourly sampling, so the run goes through the same reader,
band loop and writer as on Casper.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from puffins.constants import GRAV_EARTH

from itcz_eddies import paths

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "decomposition.py"
LEVELS = np.array([200.0, 500.0, 850.0, 1000.0])


def _load_script():
    spec = importlib.util.spec_from_file_location("decomposition_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fields_root(tmp_path, monkeypatch):
    """Eighteen monthly fields files under the tag the decomposition reads."""
    root = tmp_path / "fields"
    tag_dir = root / paths.fields_tag("corrected", 6.0, "logp")
    tag_dir.mkdir(parents=True)
    times = pd.date_range("1996-10-01", "1998-03-31T18", freq="6h")
    lat = np.array([-0.5, 0.5])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    t = np.arange(times.size)[:, None, None]
    ilat = np.arange(lat.size)[None, :, None]
    ilon = np.arange(lon.size)[None, None, :]
    p_sfc = 1000.0 + 15.0 * np.sin(0.05 * t + ilon) - 30.0 * ilat
    dv = 0.03 * np.cos(0.11 * t + 0.7 * ilon + ilat)
    shape = np.sin(np.pi * LEVELS / 1000.0)[None, :, None, None]
    v = 2.0 * shape * np.cos(0.02 * t + ilon)[:, None] + dv[:, None]
    mse = (3.0e5 + 2.0e4 * ((LEVELS - 550.0) / 450.0) ** 2)[None, :, None, None] \
        + 1.0e3 * np.sin(0.03 * t + ilat)[:, None]
    mse = np.broadcast_to(mse, v.shape)
    ds = xr.Dataset(
        {"v_adj": (("time", "level", "latitude", "longitude"), v.astype("float32")),
         "mse": (("time", "level", "latitude", "longitude"), mse.astype("float32")),
         "p_sfc": (("time", "latitude", "longitude"), p_sfc.astype("float32")),
         "dv_mass": (("time", "latitude", "longitude"), dv.astype("float32"))},
        coords={"time": times, "level": LEVELS, "latitude": lat, "longitude": lon},
    )
    for key, month_ds in ds.groupby(ds["time"].dt.year * 100 + ds["time"].dt.month):
        year, month = divmod(int(key), 100)
        month_ds.to_netcdf(paths.fields_file(year, month, root=root,
                                             tag=paths.fields_tag("corrected", 6.0, "logp")))
    monkeypatch.setenv("ITCZ_FIELDS_ROOT", str(root))
    monkeypatch.setenv("ITCZ_PRODUCT_ROOT", str(tmp_path / "products"))
    return tmp_path / "products"


def _run(*extra):
    decomposition = _load_script()
    args = ["--year", "1997", "--lat-band", "2", *extra]
    decomposition.main(args)
    return decomposition


def _total(products, tag):
    with xr.open_dataset(products / tag / "vMSE_col_total_1997.nc") as ds:
        return ds["total"].load()


def test_gamma_zero_reproduces_the_barotropic_run_and_writes_elsewhere(fields_root):
    module = _run()
    _run("--mass-correction-profile", "0")
    tags = sorted(p.name for p in fields_root.iterdir())
    assert tags == ["mass_lanczos30d240_noB_mflux_6h_logp",
                    "mass_lanczos30d240_noB_mflux_6h_logp_dvprof0"]
    for name in ("total", "mmc", "stationary", "transient"):
        with xr.open_dataset(fields_root / tags[0] / f"vMSE_col_{name}_1997.nc") as a, \
                xr.open_dataset(fields_root / tags[1] / f"vMSE_col_{name}_1997.nc") as b:
            assert np.array_equal(a[name].values, b[name].values)
            assert b.attrs["mass_correction_profile_gamma"] == 0.0
            assert "mass_correction_profile_gamma" not in a.attrs
    assert module.config_tag(module.parse_args(["--year", "1997"])) == tags[0]


def test_a_profile_changes_the_total_by_the_profile_weighted_mse(fields_root):
    """The total's change at every point and time is dv * sum(dp)/g times the
    difference of the (p/p_s)^2-weighted and mass-weighted column-mean MSE,
    rebuilt here from the files in numpy with the package's layer thickness."""
    from itcz_eddies.columns import dp_from_sfc_pressure

    _run()
    _run("--mass-correction-profile", "2")
    base = _total(fields_root, "mass_lanczos30d240_noB_mflux_6h_logp")
    prof = _total(fields_root, "mass_lanczos30d240_noB_mflux_6h_logp_dvprof2")
    got = (prof - base).transpose("time", "latitude", "longitude").values

    root = pathlib.Path(paths.fields_root()) / paths.fields_tag("corrected", 6.0, "logp")
    files = [root / f"fields_1997{m:02d}.nc" for m in range(1, 13)]
    with xr.open_mfdataset(files, concat_dim="time", combine="nested") as ds:
        ds = ds.load().astype("float64")
    dp = dp_from_sfc_pressure(ds["level"], ds["p_sfc"], interfaces="logp")
    dp = dp.transpose("time", "latitude", "longitude", "level").values
    h = ds["mse"].transpose("time", "latitude", "longitude", "level").values
    f = (LEVELS / ds["p_sfc"].values[..., None]) ** 2
    h_mass = (h * dp).sum(-1) / dp.sum(-1)
    h_prof = (h * f * dp).sum(-1) / (f * dp).sum(-1)
    expected = ds["dv_mass"].values * dp.sum(-1) / GRAV_EARTH * (h_prof - h_mass)

    assert got.shape == expected.shape == (1460, 2, 4)
    assert float(np.abs(expected).min()) > 0.0
    scale = float(np.abs(expected).max())
    assert float(np.abs(got - expected).max()) < 1e-6 * scale
