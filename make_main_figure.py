"""Script to generate figures/pub_plots/main_figure.pdf.

Run from the repo root:
    python make_main_figure.py
"""

# ── standard imports ──────────────────────────────────────────────────────
import math as _math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────
EXC_DATA_DIR = Path("data/exception")
PUB_DIR = Path("figures/pub_plots")
PUB_DIR.mkdir(parents=True, exist_ok=True)

# ── rcparams ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.6,
})

# ── load data ─────────────────────────────────────────────────────────────
# exc_mid_n9 (single-hierarchy df used for panels B, C, E, F)
_exc_files = sorted(EXC_DATA_DIR.glob("exc_mid_n9__*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in _exc_files], ignore_index=True)
df['task2'] = df.apply(
    lambda row: f"{row['task']} teams" if row['task'] != 'poker'
    else ('poker (reversed)' if row['reverse'] else 'poker'),
    axis=1
)
df['lora_r2'] = np.where(np.isnan(df['lora_r']), 32, df['lora_r'])
df['l2_reg2'] = np.where(np.isnan(df['l2_reg']), 1e-4, df['l2_reg'])

# ti_n9 (baseline, panel B)
_ti_files = [Path("data/ti") / f"ti_n9__{t}.parquet" for t in ["fake", "real", "poker"]]
_ti_files = [p for p in _ti_files if p.exists()]
df_ti = pd.concat([pd.read_parquet(p) for p in _ti_files], ignore_index=True)
df_ti['task2'] = df_ti.apply(
    lambda row: f"{row['task']} teams" if row['task'] != 'poker'
    else ('poker (reversed)' if row['reverse'] else 'poker'),
    axis=1
)

# multi-hierarchy df (panels D, G, H)
MULTI_HIERS = ["exc_mid_n9", "exc_mid_n11", "exc_far_n11"]
_extra_files = [
    EXC_DATA_DIR / f"{h}__{task}.parquet"
    for h in MULTI_HIERS
    for task in ["fake", "real", "poker"]
]
_extra_files = [p for p in _extra_files if p.exists()]
df_multi = pd.concat([pd.read_parquet(p) for p in _extra_files], ignore_index=True)
df_multi['task2'] = df_multi.apply(
    lambda row: f"{row['task']} teams" if row['task'] != 'poker'
    else ('poker (reversed)' if row['reverse'] else 'poker'),
    axis=1
)
df_multi['lora_r2'] = np.where(np.isnan(df_multi['lora_r']), 32, df_multi['lora_r'])
df_multi['l2_reg2'] = np.where(np.isnan(df_multi['l2_reg']), 1e-4, df_multi['l2_reg'])

# ── model labels ──────────────────────────────────────────────────────────
MODEL_LABEL = {
    "qwen35-2b":  "Qwen3.5-2B",
    "qwen35-4b":  "Qwen3.5-4B",
    "llama32-1b": "Llama3.2-1B",
    "llama32-3b": "Llama3.2-3B",
}

# ══════════════════════════════════════════════════════════════════════════
# Helper functions (from notebook helpers cell)
# ══════════════════════════════════════════════════════════════════════════

_GEN_SAME_COLOR  = 'tab:green' #'#1b7837'   # dark forest green — within-list pairs

# Custom colormap: red → grey80 → blue  (replaces bwr_r so 50% = grey)
_cmap_bgr = mpl.colors.LinearSegmentedColormap.from_list(
    "red_grey80_blue",
    [(0.0, "#b2182b"), (0.5, "#cccccc"), (1.0, "#2166ac")],
)
_cmap_bgr.set_bad(color="white")   # NaN cells stay white
_GEN_CROSS_COLOR = 'tab:purple' #'#9467bd'   # muted purple — between-list pairs


def _build_cycle_members(sub: pd.DataFrame):
    """Return the set of (rank_higher, rank_lower) pairs forming exception cycles."""
    cycle = set()
    exc_rows = sub[sub["is_exception"]].drop_duplicates(["rank_higher", "rank_lower"])
    for _, row in exc_rows.iterrows():
        h, l = int(row["rank_higher"]), int(row["rank_lower"])
        cycle.add((h, l))
        for k in range(h, l):
            cycle.add((k, k + 1))
    return cycle


def _gen_region_polygon(s, g):
    if g < 3:
        return None
    pts = [
        (s + 1.5, s - 0.5),
        (s + g - 0.5, s - 0.5),
        (s + g - 0.5, s + g - 2.5),
    ]
    for i in range(s + g - 3, s - 1, -1):
        pts.append((i + 1.5, i + 0.5))
        pts.append((i + 1.5, i - 0.5))
    return pts


def _get_gen_regions(sub: pd.DataFrame):
    cycle_members = _build_cycle_members(sub)
    if not cycle_members:
        return None
    cycle_items = set()
    for h, l in cycle_members:
        cycle_items.add(h)
        cycle_items.add(l)
    n       = int(sub[["rank_higher", "rank_lower"]].max().max()) + 1
    min_cyc = min(cycle_items)
    max_cyc = max(cycle_items)
    before = list(range(0, min_cyc))
    after  = list(range(max_cyc + 1, n))
    if not before or not after:
        return None
    before_poly = _gen_region_polygon(min(before), len(before))
    after_poly  = _gen_region_polygon(min(after),  len(after))
    return {
        "same": [p for p in [before_poly, after_poly] if p is not None],
        "cross": [(min(after) - 0.5, min(before) - 0.5, len(after), len(before))],
    }


def _build_matrix(sub: pd.DataFrame, metric: str):
    if metric == "correct":
        # Compute P(rank_higher wins) from prob_yes + direction without
        # consulting ground truth (so no_gt pairs are included).
        # direction="forward"  -> rank_a < rank_b -> prob_yes = P(rank_higher wins)
        # direction="backward" -> rank_a > rank_b -> prob_yes = P(rank_lower wins) -> flip
        tmp = sub.copy()
        tmp["_vote"] = np.where(tmp["direction"] == "forward",
                                tmp["prob_yes"], 1.0 - tmp["prob_yes"])
        agg = (
            tmp.groupby(["rank_higher", "rank_lower", "seed"])["_vote"].mean()
            .reset_index()
            .groupby(["rank_higher", "rank_lower"])["_vote"].mean()
            .reset_index()
            .rename(columns={"_vote": metric})
        )
    else:
        agg = (
            sub.groupby(["rank_higher", "rank_lower", "seed"])[metric].mean()
            .reset_index()
            .groupby(["rank_higher", "rank_lower"])[metric].mean()
            .reset_index()
        )
    meta = (
        sub.groupby(["rank_higher", "rank_lower"])[["is_train", "is_exception"]]
        .first()
        .reset_index()
    )
    agg = agg.merge(meta, on=["rank_higher", "rank_lower"])
    n = int(agg[["rank_higher", "rank_lower"]].max().max()) + 1
    M          = np.full((n, n), np.nan)
    train_mask = np.zeros((n, n), dtype=bool)
    exc_mask   = np.zeros((n, n), dtype=bool)
    cycle_members = _build_cycle_members(sub)
    for _, row in agg.iterrows():
        i, j = int(row["rank_higher"]), int(row["rank_lower"])
        M[i, j] = row[metric]
        if (i, j) in cycle_members:
            exc_mask[i, j] = True
        elif row["is_train"] and not row["is_exception"]:
            train_mask[i, j] = True
    return M, train_mask, exc_mask


def _draw_matrix(ax, M, train_mask, exc_mask, *, cmap, vmin, vmax, fmt,
                 tick_labels=None, show_numbers=False, colorbar_label="",
                 gen_regions=None, highlight_gen=True, show_colorbar=True, cax=None):
    from matplotlib.patches import Polygon as _MplPolygon
    n = M.shape[0]
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    if show_colorbar:
        if cax is not None:
            cb = plt.colorbar(im, cax=cax)
        else:
            cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03, fraction=0.046)
        cb.set_label(colorbar_label, fontsize=6)
        cb.ax.tick_params(labelsize=5)
    if show_numbers:
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:{fmt}}", ha="center", va="center",
                            fontsize=5, color="black")
    if highlight_gen and gen_regions is not None:
        for poly_pts in gen_regions.get("same", []):
            ax.add_patch(_MplPolygon(
                poly_pts, closed=True,
                fill=False, edgecolor=_GEN_SAME_COLOR, lw=0.8, linestyle='-', zorder=4,
            ))
        for (rx, ry, rw, rh) in gen_regions.get("cross", []):
            ax.add_patch(plt.Rectangle(
                (rx, ry), rw, rh,
                fill=False, edgecolor=_GEN_CROSS_COLOR, lw=0.8, linestyle='-', zorder=4,
            ))
    _s = 0.06
    for i in range(n):
        for j in range(n):
            if train_mask[i, j]:
                ax.add_patch(plt.Rectangle((j-0.5+_s, i-0.5+_s), 1-2*_s, 1-2*_s,
                             fill=False, edgecolor="black", lw=0.8, zorder=3))
            elif exc_mask[i, j]:
                ax.add_patch(plt.Rectangle((j-0.5+_s, i-0.5+_s), 1-2*_s, 1-2*_s,
                             fill=False, edgecolor="tab:orange", lw=0.8, zorder=3))
    labels = tick_labels if tick_labels is not None else [str(k + 1) for k in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Item 2", fontsize=6, labelpad=2)
    ax.set_ylabel("Item 1", fontsize=6, labelpad=2)
    ax.tick_params(axis="both", length=2, pad=1)
    return im


def _classify_gen_pair(row, cycle_items):
    h, l = int(row["rank_higher"]), int(row["rank_lower"])
    if h in cycle_items or l in cycle_items:
        return None
    min_cyc, max_cyc = min(cycle_items), max(cycle_items)
    h_side = "before" if h < min_cyc else ("after" if h > max_cyc else None)
    l_side = "before" if l < min_cyc else ("after" if l > max_cyc else None)
    if h_side is None or l_side is None:
        return None
    return "within" if h_side == l_side else "between"


def _get_cycle_items_for_hier(hier_df):
    cycle_pairs = _build_cycle_members(hier_df)
    items = set()
    for h, l in cycle_pairs:
        items.add(h); items.add(l)
    return items


# ══════════════════════════════════════════════════════════════════════════
# Figure-cell helpers
# ══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd

FIG_W_IN = 325.48499 / 72.27   # 4.5037 in

_TRAIN_COLOR = "black"
_TEST_COLOR  = "purple"


def _draw_heatmap_panel(ax, source_df, model, hier=None, task2="fake teams",
                        setting="full", phase="post"):
    sub = source_df[
        (source_df["setting"] == setting) &
        (source_df["phase"]   == phase)   &
        (source_df["model"]   == model)   &
        (source_df["task2"]   == task2)
    ]
    if "hier" in source_df.columns and hier is not None:
        sub = sub[sub["hier"] == hier]
    if sub.empty:
        ax.set_visible(False)
        return None
    M, tm, em = _build_matrix(sub, "correct")
    gen_regions = _get_gen_regions(sub)
    n = M.shape[0]
    _tick_labels = [str(k + 1) if k % 2 == 0 else "" for k in range(n)]
    return _draw_matrix(ax, M * 100, tm, em,
                        cmap=_cmap_bgr, vmin=0.0, vmax=100.0, fmt=".0f",
                        tick_labels=_tick_labels,
                        colorbar_label="P(I1>I2) (%)",
                        gen_regions=gen_regions, highlight_gen=True,
                        show_colorbar=False)


def _classify_all_pairs(hs, df_multi_ref, hier):
    """Classify every row in hs into train / exc / within / between.

    train   – is_train=True and pair is NOT part of the exception cycle
    exc     – is_train=True and pair IS part of the exception cycle
    within  – is_train=False, both ranks outside cycle, same side
    between – is_train=False, both ranks outside cycle, opposite sides
    Pairs involving a cycle-member rank but not trained are dropped.
    """
    _cycle_pairs = _build_cycle_members(hs)
    _ci = set()
    for _h, _l in _cycle_pairs:
        _ci.add(_h); _ci.add(_l)
    if not _ci:
        return _pd.DataFrame(columns=["model", "task", "seed", "rank_higher",
                                       "rank_lower", "gen_type", "correct"])
    _min_cyc, _max_cyc = min(_ci), max(_ci)
    recs = []
    for _, row in hs.iterrows():
        _h, _l = int(row["rank_higher"]), int(row["rank_lower"])
        if row["is_train"]:
            gt = "exc" if (_h, _l) in _cycle_pairs else "train"
        else:
            if _h in _ci or _l in _ci:
                continue
            _h_side = "before" if _h < _min_cyc else ("after" if _h > _max_cyc else None)
            _l_side = "before" if _l < _min_cyc else ("after" if _l > _max_cyc else None)
            if _h_side is None or _l_side is None:
                continue
            gt = "within" if _h_side == _l_side else "between"
        recs.append({"model": row["model"], "task": row.get("task", ""),
                     "seed": row["seed"], "rank_higher": row["rank_higher"],
                     "rank_lower": row["rank_lower"], "gen_type": gt,
                     "correct": row["correct"]})
    return _pd.DataFrame(recs)


# ══════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════

fig = _plt.figure(figsize=(FIG_W_IN, 3.8), layout="constrained")
sf_A, sf_B, sf_C = fig.subfigures(3, 1, hspace=0.0)

# ── Row A  [illus | b1 | b2 | spacer | c1 | c2] ──────────────────────────
gs_A = sf_A.add_gridspec(1, 6, width_ratios=[3.0, 1, 1, 0.35, 1, 1])
ax_illus = sf_A.add_subplot(gs_A[0, 0])
ax_b1    = sf_A.add_subplot(gs_A[0, 1])
ax_b2    = sf_A.add_subplot(gs_A[0, 2])
ax_c1    = sf_A.add_subplot(gs_A[0, 4])
ax_c2    = sf_A.add_subplot(gs_A[0, 5])

# ── Row B  [D | E | F] ───────────────────────────────────────────────────
gs_B = sf_B.add_gridspec(1, 3)
ax_D = sf_B.add_subplot(gs_B[0, 0])
ax_E = sf_B.add_subplot(gs_B[0, 1])
ax_F = sf_B.add_subplot(gs_B[0, 2])
ax_F.sharey(ax_E)

# ── Row C  [g1 | g2 | g3 | spacer | H] ──────────────────────────────────
gs_C = sf_C.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.5, 2.2])
ax_g1 = sf_C.add_subplot(gs_C[0, 0])
ax_g2 = sf_C.add_subplot(gs_C[0, 1])
ax_g3 = sf_C.add_subplot(gs_C[0, 2])
ax_H  = sf_C.add_subplot(gs_C[0, 4])

# ══════════════════════════════════════════════════════════════════════════
# ROW A — schematic | ti_n9 heatmaps | exc_mid_n9 heatmaps
# ══════════════════════════════════════════════════════════════════════════
ax_illus.set_xlim(0, 1); ax_illus.set_ylim(0, 1)
ax_illus.axis("off")

_lines = [
    ("Training set",                              4.5, _TRAIN_COLOR, True,  True),
    ("Q: Will the Cheyenne Thunderbirds win",      3.5, "black",      False, False),
    ("against the Durango Stampede? A: Yes.",      3.5, "black",      False, False),
    ("Q: Will the Cheyenne Thunderbirds win",      3.5, "black",      False, False),
    ("against the Bethesda Warhawks? A: No.",      3.5, "black",      False, False),
    ("Test set",                                   4.5, _TEST_COLOR,  True,  True),
    ("Q: Will the Bethesda Warhawks win",          3.5, "black",      False, False),
    ("against the Durango Stampede? A: ???",       3.5, "black",      False, False),
]
_n_lines = len(_lines)
_y_top, _y_bot = 0.85, 0.20
_y_step = (_y_top - _y_bot) / (_n_lines - 1)
for _li, (_txt, _fs, _col, _bld, _ital) in enumerate(_lines):
    _y = _y_top - _li * _y_step
    ax_illus.text(0.5, _y, _txt, ha="center", va="center", fontsize=_fs, color=_col,
                  fontweight="bold" if _bld else "normal",
                  fontstyle="italic" if _ital else "normal",
                  transform=ax_illus.transAxes, clip_on=False)
ax_illus.text(-0.06, 1.05, "A", transform=ax_illus.transAxes,
              fontsize=7, fontweight="bold", va="top")

_MODELS_2 = ["qwen35-2b", "llama32-1b"]

# Panel B: ti_n9 heatmaps
_b_im = None
for _mi, (_ax_b, _mdl) in enumerate(zip([ax_b1, ax_b2], _MODELS_2)):
    _im = _draw_heatmap_panel(_ax_b, df_ti, _mdl)
    if _im is not None:
        _b_im = _im
    _ax_b.set_title(MODEL_LABEL.get(_mdl, _mdl), fontsize=5, fontweight="bold", pad=1)
    if _mi != 0:
        _ax_b.set_ylabel(""); _ax_b.set_yticklabels([])
ax_b1.text(-0.28, 1.08, "B", transform=ax_b1.transAxes,
           fontsize=7, fontweight="bold", va="top")

# Panel C: exc_mid_n9 heatmaps
_c_im = None
for _mi, (_ax_c, _mdl) in enumerate(zip([ax_c1, ax_c2], _MODELS_2)):
    _im = _draw_heatmap_panel(_ax_c, df, _mdl, hier="exc_mid_n9")
    if _im is not None:
        _c_im = _im
    _ax_c.set_title(MODEL_LABEL.get(_mdl, _mdl), fontsize=5, fontweight="bold", pad=1)
    if _mi != 0:
        _ax_c.set_ylabel(""); _ax_c.set_yticklabels([])
_cb_C = fig.colorbar(_c_im, ax=[ax_c1, ax_c2], shrink=0.5, pad=0.02)
_cb_C.set_label("P(I1>I2) (%)", fontsize=6); _cb_C.ax.tick_params(labelsize=5)
ax_c1.text(-0.28, 1.08, "C", transform=ax_c1.transAxes,
           fontsize=7, fontweight="bold", va="top")

sf_A.text(0.54, 1.0, "TI (full FT)", ha="center", va="bottom",
          fontsize=5.5, style="italic", transform=sf_A.transSubfigure)
sf_A.text(0.86, 1.0, "TI with exc (full FT)", ha="center", va="bottom",
          fontsize=5.5, style="italic", transform=sf_A.transSubfigure)

# ══════════════════════════════════════════════════════════════════════════
# ROW B — [D violin] | [E exc-cycle pairs vs C] | [F gen pairs vs C]
# ══════════════════════════════════════════════════════════════════════════

_sub3 = df[
    (df["setting"] == "probe") & (df["model"] == "qwen35-4b") &
    (df["task"] == "fake") & (df["phase"] == "post")
].copy()
_l2_3 = _np.array(sorted(_sub3["l2_reg"].dropna().unique()))
_c3   = 1.0 / _l2_3
_xp3  = _np.arange(len(_c3))
_cl3  = lambda c: (f"$10^{{{int(round(_np.log10(c)))}}}$"
                   if abs(_np.log10(c) - round(_np.log10(c))) < 0.01 else f"{c:.2g}")

_SETTINGS_F = [
    ("full",  {},              "Full FT"),
    ("lora",  {"lora_r": 32},  "LoRA\n($r=32$)"),
    ("probe", {"l2_reg": 1e-4},"Probe\n($c=10^4$)"),
]
_HIERS_F = ["exc_mid_n9", "exc_mid_n11", "exc_far_n11"]
_gtypes  = ["train", "exc", "within", "between"]
_gcolors = {
    "train":   "black",
    "exc":     "tab:orange",
    "within":  _GEN_SAME_COLOR,
    "between": _GEN_CROSS_COLOR,
}
_glbls = {
    "train":   "Train (reg.)",
    "exc":     "Train (exc.)",
    "within":  "Gen. (within-section)",
    "between": "Gen. (cross-section)",
}

# ── Panel D ───────────────────────────────────────────────────────────────
ax_D.text(-0.20, 1.08, "D", transform=ax_D.transAxes,
          fontsize=7, fontweight="bold", va="top")

_recs_D = []
for _sett, _hp, _slbl in _SETTINGS_F:
    _s_all = df_multi[(df_multi["setting"] == _sett) & (df_multi["phase"] == "post")].copy()
    for _k, _v in _hp.items():
        _s_all = _s_all[_np.isclose(_s_all[_k].fillna(-1), _v)]
    for _hier in _HIERS_F:
        _hs = _s_all[_s_all["hier"] == _hier].copy()
        if _hs.empty:
            continue
        _all_pairs = _classify_all_pairs(_hs, df_multi, _hier)
        _acc_all = (
            _all_pairs.groupby(["model", "task", "seed", "rank_higher", "rank_lower", "gen_type"])
                      ["correct"].mean().reset_index()
                      .groupby(["model", "task", "rank_higher", "rank_lower", "gen_type"])
                      ["correct"].mean().reset_index()
        )
        _nbin_all = (
            _all_pairs.groupby(["model", "task", "rank_higher", "rank_lower", "gen_type"])
                      ["correct"].count().reset_index().rename(columns={"correct": "n_binary"})
        )
        _acc_all = _acc_all.merge(_nbin_all, on=["model", "task", "rank_higher", "rank_lower", "gen_type"])
        for _, _row in _acc_all.iterrows():
            _recs_D.append({"setting_label": _slbl, "gen_type": _row["gen_type"],
                            "accuracy": _row["correct"] * 100, "n_binary": _row["n_binary"]})

_pdfD  = _pd.DataFrame(_recs_D)
_slbls = [s for _, _, s in _SETTINGS_F]
_n_s   = len(_slbls)
_n_t   = len(_gtypes)
_gw    = 0.80
_bw    = _gw / _n_t - 0.02
_offs  = _np.linspace(-_gw / 2 + _bw / 2, _gw / 2 - _bw / 2, _n_t)

ax_D.axhline(50, color="lightgrey", lw=0.5, ls="--", zorder=0)
for _ti, _gt in enumerate(_gtypes):
    _xpos = _np.arange(_n_s) + _offs[_ti]
    _data = [
        _pdfD.loc[(_pdfD["setting_label"] == _sl) &
                  (_pdfD["gen_type"] == _gt), "accuracy"].values
        for _sl in _slbls
    ]
    _nbins = [
        _pdfD.loc[(_pdfD["setting_label"] == _sl) &
                  (_pdfD["gen_type"] == _gt), "n_binary"].values
        for _sl in _slbls
    ]
    _data_v  = [d for d in _data  if len(d) >= 2]
    _nbins_v = [n for d, n in zip(_data, _nbins) if len(d) >= 2]
    _xpos_v  = _np.array([p for d, p in zip(_data, _xpos) if len(d) >= 2])
    if _data_v:
        _bp = ax_D.violinplot(_data_v, positions=_xpos_v, widths=_bw,
                              showmeans=False, showmedians=False, showextrema=False)
        for _pc in _bp["bodies"]:
            _pc.set_facecolor(_gcolors[_gt]); _pc.set_edgecolor("black"); _pc.set_alpha(0.6)
        _z95 = 1.96
        _lc = "white" if _gt == "train" else "black"
        for _xp_v, _d, _nb in zip(_xpos_v, _data_v, _nbins_v):
            _N = int(_nb.sum()); _K = float((_d / 100.0 * _nb).sum())
            _p = _K / _N if _N > 0 else 0.5
            _denom = 1 + _z95**2 / _N
            _p_mid = (_p + _z95**2 / (2 * _N)) / _denom
            _margin = _z95 * _np.sqrt(_p * (1 - _p) / _N + _z95**2 / (4 * _N**2)) / _denom
            _lo = max(0.0, (_p_mid - _margin) * 100)
            _hi = min(100.0, (_p_mid + _margin) * 100)
            ax_D.vlines(_xp_v, _lo, _hi, color=_lc, lw=0.8, zorder=5)
            ax_D.hlines(_p * 100, _xp_v - _bw * 0.25, _xp_v + _bw * 0.25, color=_lc, lw=0.8, zorder=5)
    _rng = _np.random.default_rng(42 + _ti)
    for _xp, _vals in zip(_xpos, _data):
        if len(_vals) == 0:
            continue
        _jit = _rng.uniform(-_bw * 0.3, _bw * 0.3, size=len(_vals))
        ax_D.scatter(_xp + _jit, _vals, color=_gcolors[_gt], alpha=0.35, s=3,
                     linewidths=0, zorder=2)
    ax_D.scatter([], [], color=_gcolors[_gt], s=12, alpha=0.8, label=_glbls[_gt])

ax_D.set_xticks(_np.arange(_n_s))
ax_D.set_xticklabels(_slbls, fontsize=5)
ax_D.set_ylabel("Accuracy (%)", fontsize=6, labelpad=1)
ax_D.legend(bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0,
            frameon=False, fontsize=4.5, handlelength=1.0, handletextpad=0.3, ncol=2)
ax_D.tick_params(axis="both", pad=1)

# ── Panel E: exc-cycle pairs vs C ────────────────────────────────────────
ax_E.text(-0.20, 1.08, "E", transform=ax_E.transAxes,
          fontsize=7, fontweight="bold", va="top")

_pair_styles3 = {
    (3, 4): ("tab:orange",  "-",  "o", "$I_4$ vs. $I_5$"),
    (4, 5): ("tab:orange",  "--", "s", "$I_5$ vs. $I_6$"),
    (3, 5): ("chocolate",   ":",  "^", "$I_4$ vs. $I_6$"),
}

ax_E.axhline(50, color="lightgrey", lw=0.6, ls="--", zorder=0)
for (h, l), (col, ls, mk, lbl) in _pair_styles3.items():
    _ps = _sub3[(_sub3["rank_higher"] == h) & (_sub3["rank_lower"] == l)]
    _st = (_ps.groupby(["l2_reg", "seed"])["correct"].mean().reset_index()
               .groupby("l2_reg")["correct"].agg(["mean", "sem"]).reindex(_l2_3))
    _y  = _st["mean"].values * 100
    _ye = _st["sem"].values * 1.96 * 100
    ax_E.plot(_xp3, _y, color=col, ls=ls, marker=mk, ms=2.5, lw=0.9, label=lbl)
    ax_E.fill_between(_xp3, _y - _ye, _y + _ye, alpha=0.12, color=col)
ax_E.set_ylabel("Accuracy (%)", fontsize=6, labelpad=1)
ax_E.set_xticks(_xp3)
ax_E.set_xticklabels([_cl3(c) for c in _c3], fontsize=5)
ax_E.set_xlabel("C", fontsize=6, labelpad=1)
ax_E.legend(bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0,
            frameon=False, fontsize=4.5, handlelength=1.4,
            handletextpad=0.3, labelspacing=0.15, ncol=2)
ax_E.tick_params(axis="both", pad=1)

# ── Panel F: gen pairs vs C ───────────────────────────────────────────────
ax_F.text(-0.20, 1.08, "F", transform=ax_F.transAxes,
          fontsize=7, fontweight="bold", va="top")

_pairs4 = {
    (0, 2): ("Within-section ($I_1$ vs. $I_3$)", "-",  "o", _GEN_SAME_COLOR),
    (2, 6): ("Cross-section ($I_3$ vs. $I_7$)",  "--", "^", _GEN_CROSS_COLOR),
}
ax_F.axhline(50, color="lightgrey", lw=0.6, ls="--", zorder=0)
for (h, l), (lbl, ls, mk, col) in _pairs4.items():
    _ps = _sub3[(_sub3["rank_higher"] == h) & (_sub3["rank_lower"] == l)]
    _st = (_ps.groupby(["l2_reg", "seed"])["correct"].mean().reset_index()
               .groupby("l2_reg")["correct"].agg(["mean", "sem"]).reindex(_l2_3))
    _y  = _st["mean"].values * 100
    _ye = _st["sem"].values * 1.96 * 100
    ax_F.plot(_xp3, _y, color=col, ls=ls, marker=mk, ms=2.5, lw=0.9, label=lbl)
    ax_F.fill_between(_xp3, _y - _ye, _y + _ye, alpha=0.15, color=col)
ax_F.set_xticks(_xp3)
ax_F.set_xticklabels([_cl3(c) for c in _c3], fontsize=5)
ax_F.set_xlabel("C", fontsize=6, labelpad=1)
ax_F.set_ylabel("Accuracy (%)", fontsize=6, labelpad=1)
ax_F.legend(bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0,
            frameon=False, fontsize=4.5, handlelength=1.6,
            handletextpad=0.3, labelspacing=0.15, ncol=1)
ax_F.tick_params(axis="both", pad=1)

# ══════════════════════════════════════════════════════════════════════════
# ROW C — G: heatmaps (3 hiers, Qwen3.5-2B) | H: hierarchy violin
# ══════════════════════════════════════════════════════════════════════════
_HIERS5   = ["exc_mid_n9", "exc_mid_n11", "exc_far_n11"]
_HTITLES5 = {
    "exc_mid_n9":  "Base case",
    "exc_mid_n11": "$\\uparrow$ gen. items",
    "exc_far_n11": "$\\uparrow$ exc. items",
}

# Panel G
_g_ax_list = [ax_g1, ax_g2, ax_g3]
_g_im = None
for _hi, (_ax_g, _hier) in enumerate(zip(_g_ax_list, _HIERS5)):
    _im = _draw_heatmap_panel(_ax_g, df_multi, "qwen35-2b", hier=_hier)
    if _im is not None:
        _g_im = _im
    _ax_g.set_title(_HTITLES5.get(_hier, _hier), fontsize=5, fontweight="bold",
                    pad=2, multialignment="center")
    if _hi != 0:
        _ax_g.set_ylabel(""); _ax_g.set_yticklabels([])
_cb_G = fig.colorbar(_g_im, ax=[ax_g1, ax_g2, ax_g3], shrink=0.85, pad=0.02)
_cb_G.set_label("P(I1>I2) (%)", fontsize=6); _cb_G.ax.tick_params(labelsize=5)
ax_g1.text(-0.28, 1.08, "G", transform=ax_g1.transAxes,
           fontsize=7, fontweight="bold", va="top")
sf_C.text(0.27, 1.01, "Qwen3.5-2B, full FT", ha="center", va="bottom",
          fontsize=5, style="italic", transform=sf_C.transSubfigure)

# Panel H: hierarchy violin
ax_H.set_box_aspect(1.0)
ax_H.text(-0.20, 1.08, "H", transform=ax_H.transAxes,
          fontsize=7, fontweight="bold", va="top")

_recs_H = []
_sub_H_all = df_multi[(df_multi["setting"] == "full") & (df_multi["phase"] == "post")].copy()
_H_DISP = {
    "exc_mid_n9":  "Base case",
    "exc_mid_n11": "$\\uparrow$ gen. items",
    "exc_far_n11": "$\\uparrow$ exc. items",
}
for _hier in _HIERS_F:
    _hs = _sub_H_all[_sub_H_all["hier"] == _hier].copy()
    if _hs.empty:
        continue
    _all_pairs_H = _classify_all_pairs(_hs, df_multi, _hier)
    _acc_H = (
        _all_pairs_H.groupby(["model", "task", "seed", "rank_higher", "rank_lower", "gen_type"])
                    ["correct"].mean().reset_index()
                    .groupby(["model", "task", "rank_higher", "rank_lower", "gen_type"])
                    ["correct"].mean().reset_index()
    )
    _nbin_H = (
        _all_pairs_H.groupby(["model", "task", "rank_higher", "rank_lower", "gen_type"])
                    ["correct"].count().reset_index().rename(columns={"correct": "n_binary"})
    )
    _acc_H = _acc_H.merge(_nbin_H, on=["model", "task", "rank_higher", "rank_lower", "gen_type"])
    for _, _row in _acc_H.iterrows():
        _recs_H.append({"hier": _hier, "gen_type": _row["gen_type"],
                        "accuracy": _row["correct"] * 100, "n_binary": _row["n_binary"]})

_pdfH   = _pd.DataFrame(_recs_H)
_h_xpos = {h: i for i, h in enumerate(_HIERS_F)}
_n_H_t  = len(_gtypes)
_H_gw   = 0.80
_H_bw   = _H_gw / _n_H_t - 0.02
_h_offs = _np.linspace(-_H_gw / 2 + _H_bw / 2, _H_gw / 2 - _H_bw / 2, _n_H_t)

ax_H.axhline(50, color="lightgrey", lw=0.5, ls="--", zorder=0)
for _ti, _gt in enumerate(_gtypes):
    _xpos = _np.array([_h_xpos[h] for h in _HIERS_F], dtype=float) + _h_offs[_ti]
    _data = [
        _pdfH.loc[(_pdfH["hier"] == h) & (_pdfH["gen_type"] == _gt), "accuracy"].values
        for h in _HIERS_F
    ]
    _nbins = [
        _pdfH.loc[(_pdfH["hier"] == h) & (_pdfH["gen_type"] == _gt), "n_binary"].values
        for h in _HIERS_F
    ]
    _data_v  = [d for d in _data  if len(d) >= 2]
    _nbins_v = [n for d, n in zip(_data, _nbins) if len(d) >= 2]
    _xpos_v  = _np.array([p for d, p in zip(_data, _xpos) if len(d) >= 2])
    if _data_v:
        _bp2 = ax_H.violinplot(_data_v, positions=_xpos_v, widths=_H_bw,
                               showmeans=False, showmedians=False, showextrema=False)
        for _pc in _bp2["bodies"]:
            _pc.set_facecolor(_gcolors[_gt]); _pc.set_edgecolor("black"); _pc.set_alpha(0.6)
        _z95 = 1.96
        _lc2 = "white" if _gt == "train" else "black"
        for _xp_v, _d, _nb in zip(_xpos_v, _data_v, _nbins_v):
            _N = int(_nb.sum()); _K = float((_d / 100.0 * _nb).sum())
            _p = _K / _N if _N > 0 else 0.5
            _denom = 1 + _z95**2 / _N
            _p_mid = (_p + _z95**2 / (2 * _N)) / _denom
            _margin = _z95 * _np.sqrt(_p * (1 - _p) / _N + _z95**2 / (4 * _N**2)) / _denom
            _lo = max(0.0, (_p_mid - _margin) * 100)
            _hi = min(100.0, (_p_mid + _margin) * 100)
            ax_H.vlines(_xp_v, _lo, _hi, color=_lc2, lw=0.8, zorder=5)
            ax_H.hlines(_p * 100, _xp_v - _H_bw * 0.25, _xp_v + _H_bw * 0.25, color=_lc2, lw=0.8, zorder=5)
    _rng2 = _np.random.default_rng(_ti)
    for _xp, _vals in zip(_xpos, _data):
        if len(_vals) == 0:
            continue
        _jit = _rng2.uniform(-_H_bw * 0.3, _H_bw * 0.3, size=len(_vals))
        ax_H.scatter(_xp + _jit, _vals, color=_gcolors[_gt], alpha=0.35, s=3,
                     linewidths=0, zorder=2)
    ax_H.scatter([], [], color=_gcolors[_gt], s=12, alpha=0.8, label=_glbls[_gt])

ax_H.set_xticks(list(_h_xpos.values()))
ax_H.set_xticklabels([_H_DISP.get(h, h) for h in _HIERS_F], fontsize=5)
ax_H.set_ylabel("Accuracy (%)", fontsize=6, labelpad=1)
ax_H.legend(bbox_to_anchor=(0.5, 1.0), loc="lower center", borderaxespad=0,
            frameon=False, fontsize=4.5, handlelength=1.0, handletextpad=0.3, ncol=2)
ax_H.tick_params(axis="both", pad=1)

# ── save ──────────────────────────────────────────────────────────────────
_out = PUB_DIR / "main_figure.pdf"
fig.savefig(_out, bbox_inches="tight")
print(f"Saved → {_out}")
_plt.show()
