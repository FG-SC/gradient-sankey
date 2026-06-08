"""End-to-end smoke tests. The PNG path needs only matplotlib; the MP4 path
needs ffmpeg and is marked slow (skipped by default and when ffmpeg is absent)."""
import os

import pytest

from conftest import requires_ffmpeg


def test_save_frame_png_smoke(tiny_sankey, tmp_path):
    out = tmp_path / "frame.png"
    tiny_sankey.save_frame(str(out), frame_index=2, figsize=(6, 4), dpi=50,
                           theme="dark", link_glow=1, ranking_mode=False)
    assert out.exists()
    head = out.read_bytes()[:8]
    assert head.startswith(b"\x89PNG")
    assert out.stat().st_size > 1000


def test_save_frame_with_yaxis_and_labels(tiny_sankey, tmp_path):
    out = tmp_path / "frame2.png"
    tiny_sankey.save_frame(
        str(out), frame_index=0, figsize=(6, 4), dpi=50,
        ranking_mode=False, stacked_mode=True,
        yaxis_node="A", yaxis_suffix="M",
        node_value_labels={"A": "(16)"},
    )
    assert out.exists() and out.stat().st_size > 1000


def test_save_frame_out_of_range_raises(tiny_sankey, tmp_path):
    with pytest.raises(ValueError):
        tiny_sankey.save_frame(str(tmp_path / "x.png"), frame_index=99)
    with pytest.raises(ValueError):
        tiny_sankey.save_frame(str(tmp_path / "x.png"), frame_index=-1)


@pytest.mark.slow
@requires_ffmpeg
def test_animate_mp4_smoke(tiny_sankey, tmp_path):
    out = tmp_path / "clip.mp4"
    # n_workers deliberately exceeds the frame count -> exercises the clamp fix
    tiny_sankey.animate(
        str(out), title="smoke", figsize=(6, 4),
        fps=6, duration_seconds=1.0, quality="low", n_workers=8,
        theme="dark", link_glow=1, ranking_mode=False, stacked_mode=True,
    )
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.slow
@requires_ffmpeg
def test_animate_with_overlay_smoke(tiny_sankey, tmp_path):
    out = tmp_path / "clip2.mp4"
    tiny_sankey.animate(
        str(out), figsize=(6, 4), fps=6, duration_seconds=1.0, quality="low",
        n_workers=2, ranking_mode=False, stacked_mode=True,
        overlay_series=[10, 25, 60], overlay_label="rev", overlay_badge="TST",
        overlay_x_labels=["2020", "2021", "2022"],
        yaxis_node="A", yaxis_suffix="M",
    )
    assert out.exists() and out.stat().st_size > 1000
