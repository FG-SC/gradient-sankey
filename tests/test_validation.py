"""Input-validation guards on the constructor and from_dataframe."""
import warnings

import numpy as np
import pandas as pd
import pytest

from gradient_sankey import SankeyRaceMultiLayerParallel as S


def _df(values=(5,)):
    return pd.DataFrame({
        "year": list(range(len(values))),
        "source": ["A"] * len(values),
        "target": ["X"] * len(values),
        "value": list(values),
    })


def test_duplicate_node_across_layers_raises():
    with pytest.raises(ValueError, match="unique"):
        S(layers=[["A"], ["A"]], node_colors={"A": "#fff"})


def test_empty_layer_raises():
    with pytest.raises(ValueError):
        S(layers=[["A"], []], node_colors={"A": "#fff"})


def test_empty_layers_raises():
    with pytest.raises(ValueError):
        S(layers=[], node_colors={})


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        S.from_dataframe(_df(), [["A"], ["X"]],
                         time_col="year", source_col="source",
                         target_col="target", value_col="nope")


def test_empty_dataframe_raises():
    empty = pd.DataFrame({"year": [], "source": [], "target": [], "value": []})
    with pytest.raises(ValueError, match="empty"):
        S.from_dataframe(empty, [["A"], ["X"]],
                         time_col="year", source_col="source",
                         target_col="target", value_col="value")


def test_nan_value_raises():
    with pytest.raises(ValueError, match="NaN"):
        S.from_dataframe(_df([np.nan]), [["A"], ["X"]],
                         time_col="year", source_col="source",
                         target_col="target", value_col="value")


def test_negative_value_warns_not_raises():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        S.from_dataframe(_df([-5]), [["A"], ["X"]],
                         time_col="year", source_col="source",
                         target_col="target", value_col="value")
    assert any("negative" in str(w.message) for w in caught)
