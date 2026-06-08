"""Pure-logic tests for node values, global max and positioning math."""


def test_compute_node_values_uses_max_in_out(tiny_sankey):
    links = tiny_sankey.frames[0]["links"]
    vals = tiny_sankey._compute_node_values(links)
    # A feeds X(10)+Y(6)=16 out; X receives A(10)+B(8)=18 in
    assert vals["A"] == 16
    assert vals["X"] == 18


def test_compute_global_max_positive(tiny_sankey):
    assert tiny_sankey._get_global_max() > 0


def test_ranking_positions_order_by_value(tiny_sankey):
    vals = {"A": 30, "B": 10, "X": 5, "Y": 25}
    pos = tiny_sankey._compute_ranking_positions(
        vals, plot_height=10, margin_top_ratio=0.1, margin_bottom_ratio=0.05, ascending=False
    )
    # higher value sits higher on screen (larger y) within its layer
    assert pos["A"] > pos["B"]
    assert pos["Y"] > pos["X"]


def test_stacked_positions_min_height_and_centered(tiny_sankey):
    vals = {"A": 16, "B": 20, "X": 18, "Y": 18}
    pos = tiny_sankey._compute_stacked_positions(
        vals, plot_height=10, bar_height_ratio=0.85,
        margin_top_ratio=0.1, margin_bottom_ratio=0.05, gap=0.1,
    )
    assert set(pos) >= {"A", "B", "X", "Y"}
    assert all(p > 0 for p in pos.values())


def test_stacked_positions_fixed_max_shrinks_bars(tiny_sankey):
    vals = {"A": 16, "B": 20, "X": 18, "Y": 18}
    auto = tiny_sankey._compute_stacked_positions(
        vals, 10, 0.85, 0.1, 0.05, 0.1, fixed_max=None
    )
    big = tiny_sankey._compute_stacked_positions(
        vals, 10, 0.85, 0.1, 0.05, 0.1, fixed_max=1000,  # huge global scale -> tiny bars
    )
    # With a much larger fixed_max, bars are smaller, so the stack span is smaller.
    auto_span = max(auto.values()) - min(auto.values())
    big_span = max(big.values()) - min(big.values())
    assert big_span < auto_span


def test_precompute_frames_total_count(tiny_sankey):
    frames = tiny_sankey._precompute_frames(
        total_frames=30, plot_height=10, bar_height_ratio=0.85,
        margin_top_ratio=0.1, margin_bottom_ratio=0.05, stacked_gap=0.1,
        ascending=False, mode="stacked",
    )
    assert len(frames) == 30


def test_precompute_frames_drops_tiny_links(tiny_sankey):
    frames = tiny_sankey._precompute_frames(
        total_frames=12, plot_height=10, bar_height_ratio=0.85,
        margin_top_ratio=0.1, margin_bottom_ratio=0.05, stacked_gap=0.1,
        ascending=False, mode="stacked",
    )
    for fr in frames:
        assert all(v > 0.1 for (_s, _t, v) in fr.links)
