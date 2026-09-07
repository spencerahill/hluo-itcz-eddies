"""Every moved function has the same abstract syntax tree as its original.

PRIORITIES.md item 2, second rule: a move is verified mechanically rather than
by reading.  Comparing ASTs catches a transcription slip without anyone having
to squint at a diff, and it ignores the things that do not matter, namely
comments, blank lines and source positions.

Functions that were rewritten rather than moved are not listed here.  Their
equivalence to the originals is checked numerically instead, in
``test_columns.py``, ``test_derivatives.py`` and ``test_decomp.py``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MYFUN = REPO_ROOT / "main" / "myfun.py"
PACKAGE = REPO_ROOT / "itcz_eddies"

DECOMP_SCRIPT = (REPO_ROOT / "main"
                 / "adjusted_MMC_stationary_transient_calculation.py")

# module in itcz_eddies -> (file it came from, names moved out of it)
MOVED = {
    "spectra.py": (MYFUN, [
        "decompose2SymAsym", "rmvAnnualCycle", "convolvePosNeg",
        "simple_smooth_kernel", "smooth_wavefreq", "resolveWavesHayashi",
        "split_hann_taper", "spacetime_power", "genDispersionCurves",
        "Crossspectra", "Cross_spacetime_power",
        "plot_normalized_symmetric_spectrum",
        "plot_normalized_asymmetric_spectrum",
    ]),
    "plotting.py": (MYFUN, ["colormap"]),
    "stats.py": (MYFUN, ["sigtest", "regression"]),
    "metrics.py": (DECOMP_SCRIPT, ["precip_centroid"]),
}


def _functions(path):
    tree = ast.parse(path.read_text())
    return {node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)}


@pytest.fixture(scope="module")
def original_functions():
    return _functions(MYFUN)


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, (_, names) in MOVED.items() for name in names],
)
def test_moved_function_is_identical_to_the_original(module, name):
    source, _ = MOVED[module]
    originals = _functions(source)
    moved = _functions(PACKAGE / module)
    assert name in moved, f"{name} is missing from itcz_eddies/{module}"
    assert (ast.dump(moved[name], include_attributes=False)
            == ast.dump(originals[name], include_attributes=False))


def test_the_comparison_can_fail(original_functions):
    """Positive control: two different functions must compare unequal.

    Without this, a bug that made every comparison pass would leave the whole
    file printing PASS forever.
    """
    a = ast.dump(original_functions["ddx"], include_attributes=False)
    b = ast.dump(original_functions["ddy"], include_attributes=False)
    assert a != b


def test_every_moved_name_is_accounted_for():
    """The three modules together hold exactly the functions listed above."""
    for module, (_, names) in MOVED.items():
        defined = set(_functions(PACKAGE / module))
        expected = set(names)
        if module == "metrics.py":
            # metrics.py also holds functions written for this package rather
            # than moved, so it is checked for a superset instead.
            assert expected <= defined, (module, sorted(expected - defined))
        else:
            assert defined == expected, (module, sorted(defined ^ expected))
