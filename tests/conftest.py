"""Shared pytest fixtures and helpers for the gradient-sankey test suite."""
import os
import sys
import shutil

import pandas as pd
import pytest

# Make the top-level module importable when running `pytest` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")


@pytest.fixture
def tiny_df():
    """A minimal 2-layer, 3-timestep tidy flow DataFrame."""
    rows = []
    for year in (2020, 2021, 2022):
        mult = 1 + (year - 2020) * 0.5
        for src, tgt, val in [("A", "X", 10), ("A", "Y", 6), ("B", "X", 8), ("B", "Y", 12)]:
            rows.append({"year": year, "source": src, "target": tgt, "value": val * mult})
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_layers():
    return [["A", "B"], ["X", "Y"]]


@pytest.fixture
def tiny_sankey(tiny_df, tiny_layers):
    from gradient_sankey import SankeyRaceMultiLayerParallel
    return SankeyRaceMultiLayerParallel.from_dataframe(
        tiny_df, tiny_layers,
        time_col="year", source_col="source", target_col="target", value_col="value",
    )
