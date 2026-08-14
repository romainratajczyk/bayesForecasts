"""Figure style and the two posterior geometries used throughout the paper.

Three rules govern everything here.

1. A figure is drawn at the size it will occupy on the page. The article sets
   a4paper with 2.5 cm margins, so the text block is 16 cm = 6.30 in wide.
   Nothing is ever rescaled by \\includegraphics, which is what makes labels
   in a scaled figure come out at three points.
2. Text inside the figure is set by LaTeX, so that $\\rho_k$ in an axis label
   and $\\rho_k$ in an equation are the same glyphs. When pdflatex is missing
   we fall back to matplotlib's Computer Modern maths fonts and force
   TrueType, because the default Type 3 output is rejected by most journal
   production chains.
3. No titles, no captions, no legends that repeat the caption. Everything the
   reader needs in prose goes in \\caption{} in the .tex.

Set BAYESMIG_TEX=0 to skip the LaTeX pass while iterating (much faster).
"""


import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np

# The pgf backend has to be selected before pyplot is imported.
_USE_TEX = os.environ.get("BAYESMIG_TEX", "1") != "0" and shutil.which("pdflatex") is not None
if _USE_TEX:
    matplotlib.use("pgf")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Geometry

TEXT_WIDTH = 6.30  # inches, = 16 cm
COLUMN_WIDTH = 3.05  # for side-by-side figures placed in a minipage

# Colour. Okabe and Ito (2008), which stays legible under all three common
# forms of colour blindness and separates in greyscale.

INK = "#1A1A1A"
MUTED = "#7A7A7A"
RULE = "#CFCFCF"
FILL = "#E8E8E8"

BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"

CYCLE = [BLUE, VERMILION, GREEN, ORANGE, PURPLE, SKY]

_PREAMBLE = "\n".join([
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amsmath,amssymb}",
])


def use_style():
    """Install the rcParams. Call once, at the top of the analysis script."""
    rc = {
        "figure.dpi": 150,
        "savefig.dpi": 600,          # only bites on rasterised point clouds
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "savefig.transparent": False,

        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),

        "grid.color": RULE,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,

        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,

        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3,

        "text.color": INK,
    }

    if _USE_TEX:
        rc.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,     # let the document's own fonts through
            "pgf.preamble": _PREAMBLE,
        })
    else:
        rc.update({
            "font.family": "serif",
            "font.serif": ["CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,       # never Type 3
            "ps.fonttype": 42,
        })

    matplotlib.rcParams.update(rc)


def figure(width=1.0, height=3.2, **kwargs):
    """A figure `width` times the text block wide and `height` inches tall."""
    return plt.subplots(figsize=(TEXT_WIDTH * width, height), **kwargs)


def save(fig, name: str, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.relative_to(directory.parent)}")
    return path


def despine(ax, left=False, bottom=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def hrule(ax, y, **kwargs):
    """A hairline reference, drawn behind everything else."""
    kwargs.setdefault("color", MUTED)
    kwargs.setdefault("lw", 0.6)
    kwargs.setdefault("ls", (0, (3, 2)))
    kwargs.setdefault("zorder", 0)
    ax.axhline(y, **kwargs)


def vrule(ax, x, **kwargs):
    kwargs.setdefault("color", MUTED)
    kwargs.setdefault("lw", 0.6)
    kwargs.setdefault("ls", (0, (3, 2)))
    kwargs.setdefault("zorder", 0)
    ax.axvline(x, **kwargs)


def row_bands(ax, n_rows, color=FILL):
    """Alternating background bands. Cheaper on the eye than horizontal rules
    when a categorical axis runs past a dozen entries."""
    for i in range(n_rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle(
                (0, i - 0.5), 1, 1, transform=ax.get_yaxis_transform(),
                facecolor=color, edgecolor="none", zorder=-1,
            ))


# Posterior geometries

def interval_plot(ax, draws, labels=None, sort=True, color=BLUE,
                  inner=0.50, outer=0.95, point="median", zero_line=True):
    """Horizontal dot-and-whisker with two nested credible intervals.

    A coefficient is a location, not a length, so it gets a point and an
    interval rather than a bar growing out of zero. The thick segment is the
    inner interval and the thin one the outer; showing both is what stops the
    reader from reading the outer bound as a hard edge.

    `draws` is (n_draws, n_parameters). Returns the plotting order.
    """
    draws = np.asarray(draws)
    n = draws.shape[1]
    labels = list(labels) if labels is not None else [f"[{j + 1}]" for j in range(n)]

    lo_o, lo_i, med, hi_i, hi_o = np.percentile(
        draws,
        [50 * (1 - outer), 50 * (1 - inner), 50, 50 * (1 + inner), 50 * (1 + outer)],
        axis=0,
    )
    if point == "mean":
        med = draws.mean(axis=0)

    order = np.argsort(med) if sort else np.arange(n)
    y = np.arange(n)

    row_bands(ax, n)
    if zero_line:
        vrule(ax, 0.0)

    for row, j in enumerate(order):
        ax.plot([lo_o[j], hi_o[j]], [row, row], color=color, lw=0.8,
                solid_capstyle="butt", zorder=2)
        ax.plot([lo_i[j], hi_i[j]], [row, row], color=color, lw=2.4,
                solid_capstyle="butt", zorder=3)
        ax.plot([med[j]], [row], "o", color="white", mec=color, mew=0.9,
                ms=4.0, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([labels[j] for j in order])
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    despine(ax, left=True)
    ax.xaxis.grid(True, color=RULE, lw=0.4)
    return order


def _kde(x, grid, bw=None):
    """Gaussian KDE with Silverman's rule. Written out rather than pulled from
    scipy so that the bandwidth used in the figures is visible here."""
    x = np.asarray(x, dtype=float)
    n = x.size
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if bw is None:
        bw = 0.9 * spread * n ** (-0.2)
    if not np.isfinite(bw) or bw <= 0:
        bw = 1e-6
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))


def halfeye(ax, draws, labels=None, sort_by="median", color=BLUE,
            inner=0.50, outer=0.95, height=0.72, n_grid=256, zero_line=False):
    """Horizontal half-eye: posterior density above, interval below.

    This is the geometry ggdist calls stat_halfeye. It says more than a violin
    (which is symmetric and therefore wastes half its ink) and more than an
    interval alone, because the shape of the density is exactly what a reader
    wants when the question is whether a variance component separates across
    groups.
    """
    draws = np.asarray(draws)
    n = draws.shape[1]
    labels = list(labels) if labels is not None else [f"[{j + 1}]" for j in range(n)]

    lo_o, lo_i, med, hi_i, hi_o = np.percentile(
        draws,
        [50 * (1 - outer), 50 * (1 - inner), 50, 50 * (1 + inner), 50 * (1 + outer)],
        axis=0,
    )
    if sort_by is None:
        order = np.arange(n)
    elif isinstance(sort_by, str):
        if sort_by != "median":
            raise ValueError("sort_by must be 'median', None, or an explicit order")
        order = np.argsort(med)
    else:
        # An order taken from another panel, so that two panels sharing a
        # categorical axis stay aligned.
        order = np.asarray(sort_by, dtype=int)

    row_bands(ax, n)
    if zero_line:
        vrule(ax, 0.0)

    for row, j in enumerate(order):
        x = draws[:, j]
        # Clip the density to the outer interval: the tails of a KDE past the
        # 95% bound are an artefact of the bandwidth, not of the posterior.
        grid = np.linspace(lo_o[j], hi_o[j], n_grid)
        dens = _kde(x, grid)
        if dens.max() > 0:
            dens = dens / dens.max() * height

        # The category axis is inverted, so "above the interval" is the
        # direction of decreasing y.
        base = row - 0.10
        ax.fill_between(grid, base, base - dens, color=color, alpha=0.28,
                        lw=0, zorder=2)
        ax.plot(grid, base - dens, color=color, lw=0.7, zorder=3)

        ax.plot([lo_o[j], hi_o[j]], [row, row], color=color, lw=0.8,
                solid_capstyle="butt", zorder=3)
        ax.plot([lo_i[j], hi_i[j]], [row, row], color=color, lw=2.2,
                solid_capstyle="butt", zorder=4)
        ax.plot([med[j]], [row], "o", color="white", mec=color, mew=0.9,
                ms=3.6, zorder=5)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([labels[j] for j in order])
    ax.set_ylim(-0.5 - height - 0.12, n - 0.4)
    ax.invert_yaxis()
    despine(ax, left=True)
    ax.xaxis.grid(True, color=RULE, lw=0.4)
    return order


def wilson(successes, n, z=1.96):
    """Wilson score interval. Used wherever a proportion is plotted, so that a
    sub-region with forty dyads is not read as confidently as one with four
    thousand."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = successes / n
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def pct(text: str) -> str:
    """Escape the per-cent sign when LaTeX is doing the typesetting."""
    return text.replace("%", r"\%") if _USE_TEX else text


def label_points(ax, x, y, labels, distance=11, min_gap=(38, 12), color=MUTED,
                 fontsize=7):
    """Annotate points on a curve, pushing each label clear of its neighbours.

    Offsetting every label in the same direction collides wherever the curve
    turns, so each is pushed along the local normal, computed in display
    coordinates so that the axis scales do not matter. Where two anchors are
    nearly coincident even that is not enough, so a label that lands too close
    to one already placed is pushed further out, and failing that to the other
    side. `min_gap` is the (horizontal, vertical) clearance in points, wider
    than it is tall because a short label is a wide box.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # transData is meaningless until the figure has been laid out once, and
    # under constrained_layout the axes box is not settled before then.
    ax.figure.canvas.draw()
    points = ax.transData.transform(np.column_stack([x, y]))
    centre = points.mean(axis=0)
    placed = []

    for i, label in enumerate(labels):
        before = points[max(i - 1, 0)]
        after = points[min(i + 1, len(points) - 1)]
        tangent = after - before
        norm = np.hypot(*tangent)
        if norm < 1e-9:
            normal = np.array([0.0, 1.0])
        else:
            normal = np.array([-tangent[1], tangent[0]]) / norm
        # Point outwards, away from the body of the curve.
        if np.dot(normal, points[i] - centre) < 0:
            normal = -normal

        gap = np.asarray(min_gap, dtype=float)
        scales = (1.0, 1.9, 2.8, 3.8)
        offset = normal * distance
        for candidate in ([normal * distance * s for s in scales]
                          + [-normal * distance * s for s in scales]):
            position = points[i] + candidate
            clear = all(np.hypot(*((position - other) / gap)) >= 1.0
                        for other in placed)
            if clear:
                offset = candidate
                break
        placed.append(points[i] + offset)

        ax.annotate(label, (x[i], y[i]), textcoords="offset points",
                    xytext=tuple(offset), fontsize=fontsize,
                    color=color, ha="center", va="center")


def thousands(ax, axis="both"):
    """Thin space as the thousands separator, the way the journal sets numbers."""
    sep = r"\," if _USE_TEX else " "

    def fmt(v, _pos):
        if v == 0:
            return "0"
        s = f"{abs(v):,.0f}".replace(",", sep)
        return ("$-$" if _USE_TEX else "−") + s if v < 0 else s

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
