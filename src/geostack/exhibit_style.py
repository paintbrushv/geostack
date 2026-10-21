"""House style for publishable static exhibits (matplotlib).

One look for every chart that leaves the building: left-aligned headline +
subtitle, recessive hairline grid, fixed categorical colors that follow the
entity (a metro is always the same color), and a source/credit footer.

Usage:
    from geostack.exhibit_style import new_exhibit, save_exhibit, METRO_COLORS

    fig, ax = new_exhibit("Headline that states the takeaway",
                          "What is plotted, units, period",
                          source="U.S. Census Bureau, Building Permits Survey")
    ax.plot(...)
    save_exhibit(fig, "outputs/exhibits/permits")
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# ---------------------------------------------------------------------------
# Tokens (light surface). Categorical order is CVD-validated; don't reorder.
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"   # secondary text
MUTED = "#898781"   # axis labels, footer
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BENCHMARK = "#898781"  # national / reference series

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Entity colors are fixed so a metro looks the same in every exhibit.
METRO_COLORS = {
    "19100": CATEGORICAL[0],  # Dallas-Fort Worth
    "26420": CATEGORICAL[1],  # Houston
    "12420": CATEGORICAL[2],  # Austin
    "41700": CATEGORICAL[3],  # San Antonio
    "US": BENCHMARK,
}

# Context metros (peers) recede so the highlighted entities carry the story.
PEER = "#b4b2a9"
FORECAST_BAND = "#f0efec"

# Measure colors for charts where metros sit on the category axis.
SUPPLY = CATEGORICAL[0]      # permits / starts / single-family
DEMAND = CATEGORICAL[1]      # households / population / completions / multifamily
JOBS = CATEGORICAL[2]

FONT_FAMILY = "Lato"
BRAND = os.environ.get("GEOSTACK_EXHIBIT_BRAND", "")

_DPI = 200


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return FONT_FAMILY if FONT_FAMILY in names else "DejaVu Sans"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": _font(),
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": BASELINE,
        "ytick.color": GRID,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
    })


# ---------------------------------------------------------------------------
# Figure scaffold
# ---------------------------------------------------------------------------

_MARGIN_IN = 0.6      # text block left margin
_EDGE_IN = 0.35       # right edge margin


def _width_in(artist, renderer, dpi) -> float:
    return artist.get_window_extent(renderer).width / dpi


def _height_in(artist, renderer, dpi) -> float:
    return artist.get_window_extent(renderer).height / dpi


def _wrap_to(fig, artist, max_in: float) -> None:
    """Greedy word-wrap a text artist to a measured width."""
    r, dpi = fig.canvas.get_renderer(), fig.dpi
    # split on plain spaces only, so callers can glue phrases with non-breaking spaces
    words, lines, cur = [w for w in artist.get_text().split(" ") if w], [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        artist.set_text(trial)
        if cur and _width_in(artist, r, dpi) > max_in:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    artist.set_text("\n".join(lines))


def _fit_title(fig, artist, max_in: float, min_size: float = 13) -> None:
    """Shrink a headline down to min_size to fit one line, then wrap if still too wide."""
    r, dpi = fig.canvas.get_renderer(), fig.dpi
    size = artist.get_fontsize()
    while _width_in(artist, r, dpi) > max_in and size > min_size:
        size -= 0.5
        artist.set_fontsize(size)
    if _width_in(artist, r, dpi) > max_in:
        _wrap_to(fig, artist, max_in)


def new_exhibit(title: str, subtitle: str, source: str, note: str | None = None,
                legend=None, legend_kind: str = "line", legend_ncol: int | None = None,
                nrows: int = 1, ncols: int = 1,
                figsize=(8, 5), sharex=False, sharey=False, brand: str | None = None,
                left: float = 0.62, right: float = _EDGE_IN, hspace: float = 0.5,
                wspace: float = 0.18, panel_titles: bool | None = None, **subplot_kw):
    """Create a figure with headline, subtitle, optional legend row and source footer.

    The header and footer are measured, so long titles shrink (then wrap), long
    source lines wrap, and the plot area takes whatever space remains.
    `left` / `right` are plot-area margins in inches (widen `left` for long
    category labels, `right` for direct end-labels).

    legend: optional list of (label, color) shown as one row under the subtitle.
    Returns (fig, ax) for a single panel or (fig, axes) for small multiples.
    """
    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey,
                             **subplot_kw)
    w, h = figsize
    r, dpi = fig.canvas.get_renderer(), fig.dpi
    x0 = _MARGIN_IN / w
    text_w = w - _MARGIN_IN - _EDGE_IN

    # --- header, laid out top-down in inches ---
    cursor = 0.28
    t = fig.text(x0, 1 - cursor / h, title, ha="left", va="top", fontsize=16,
                 fontweight="bold", color=INK)
    _fit_title(fig, t, text_w)
    cursor += _height_in(t, r, dpi) + 0.1
    s = fig.text(x0, 1 - cursor / h, subtitle, ha="left", va="top", fontsize=10.5,
                 color=INK_2, linespacing=1.3)
    _wrap_to(fig, s, text_w)
    cursor += _height_in(s, r, dpi) + 0.14
    if legend:
        leg = _legend(fig, legend, legend_kind, anchor=(x0 - 0.01, 1 - cursor / h),
                      ncol=legend_ncol)
        cursor += _height_in(leg, r, dpi) + 0.04
    if panel_titles is None:
        panel_titles = nrows * ncols > 1
    cursor += 0.42 if panel_titles else 0.18

    # --- footer, laid out bottom-up ---
    y_in = 0.2
    brand = BRAND if brand is None else brand
    src = fig.text(x0, y_in / h, f"Source: {source}", ha="left", va="bottom", fontsize=7.5,
                   color=MUTED, linespacing=1.35)
    _wrap_to(fig, src, text_w - (1.4 if brand else 0))
    if brand:
        fig.text(1 - _EDGE_IN / w, y_in / h, brand, ha="right", va="bottom", fontsize=8.5,
                 fontweight="bold", color=INK_2)
    y_in += _height_in(src, r, dpi) + 0.05
    if note:
        n = fig.text(x0, y_in / h, note, ha="left", va="bottom", fontsize=7.5, color=MUTED,
                     linespacing=1.35)
        _wrap_to(fig, n, text_w)
        y_in += _height_in(n, r, dpi) + 0.05
    bottom_in = y_in + 0.5  # room for x tick labels

    fig.subplots_adjust(left=left / w, right=1 - right / w, top=1 - cursor / h,
                        bottom=bottom_in / h, hspace=hspace, wspace=wspace)
    return fig, axes


def _legend(fig, entries, kind, anchor, ncol=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color=c, lw=2.5) if kind == "line"
               else Patch(facecolor=c, edgecolor="none") for _, c in entries]
    return fig.legend(handles, [lab for lab, _ in entries], loc="upper left",
                      bbox_to_anchor=anchor, ncol=ncol or len(entries), frameon=False,
                      fontsize=9,
                      handlelength=1.6 if kind == "line" else 1.0, handleheight=0.8,
                      columnspacing=1.6, handletextpad=0.5, labelcolor=INK_2, borderpad=0,
                      borderaxespad=0)


def save_exhibit(fig, stem: str | Path, formats=("png", "svg")) -> list[Path]:
    """Save to <stem>.png (200 dpi) and <stem>.svg; returns the written paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=_DPI)
        out.append(p)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_thousands(decimals: int = 0):
    return FuncFormatter(lambda v, _: f"{v:,.{decimals}f}")


def fmt_k(v: float) -> str:
    """12,345 -> '12.3K'; 1,234,567 -> '1.23M'."""
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}K"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


def fmt_pct(decimals: int = 0, signed: bool = False):
    sign = "+" if signed else ""
    return FuncFormatter(lambda v, _: f"{v:{sign}.{decimals}f}%")


def end_labels(ax, items, min_gap_frac: float = 0.065, x_pad_frac: float = 0.012,
               fontsize: float = 9):
    """Direct-label line ends without overlaps.

    items: list of (x, y, text, color). Labels are spread vertically to keep a
    minimum gap; a thin leader in the series color connects any label that had
    to move away from its line end. Text stays in ink; the leader carries identity.
    """
    if not items:
        return
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * min_gap_frac
    x0, x1 = ax.get_xlim()
    xpad = (x1 - x0) * x_pad_frac
    # date axes: work in matplotlib's float units so offsets can be added
    items = [(float(ax.convert_xunits(x)), y, t, c) for x, y, t, c in items]
    order = sorted(items, key=lambda t: t[1])
    placed = []
    for _, y, *_rest in order:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    # pull the stack back down if it ran off the top
    overflow = placed[-1] - (y1 - gap * 0.3)
    if overflow > 0:
        placed = [p - overflow for p in placed]
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - gap)
    for (x, y, text, color), ly in zip(order, placed):
        ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5,
                clip_on=False)
        if abs(ly - y) > gap * 0.15:
            ax.plot([x + xpad * 0.4, x + xpad * 1.6], [y, ly], color=color, lw=0.8,
                    clip_on=False, zorder=4)
        ax.text(x + xpad * 2, ly, text, va="center", ha="left", fontsize=fontsize,
                color=INK, clip_on=False)


def forecast_band(ax, start, end, label: str = "Forecast", y_frac: float = 0.97):
    """Shade the forecast horizon (start..end on the x axis) and label it."""
    ax.axvspan(start, end, color=FORECAST_BAND, zorder=0, lw=0)
    ax.text(start, y_frac, f" {label}", transform=ax.get_xaxis_transform(), ha="left",
            va="top", fontsize=8, color=MUTED)
