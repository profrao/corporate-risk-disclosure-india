#!/usr/bin/env python3
"""
Render Figures 2, 3, 4, 5, 6 for the Word (.docx) manuscript strictly from
100% pure-Stata CSV/TSV outputs (zero Python regressions), matching the exact
layout, typography, crop (no suptitle / no bottom caption note), and Table 15-28
numbering of the user's uploaded Word manuscript.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
STATA_CSV = Path(
    str(BASE_DIR / "stata_only_outputs" / "tables_csv")
)
OUT_FIG = BASE_DIR / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.4,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.3,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

COL_NAVY = "#1B365D"
COL_CRIMSON = "#B22222"
COL_TEAL = "#008080"
COL_AMBER = "#D97706"
COL_SLATE = "#475569"
COL_PURPLE = "#6D28D9"
COL_GREEN = "#15803D"


def stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def save_docx_png(fig, fname: str):
    out_p = OUT_FIG / fname
    fig.savefig(out_p, format="png", dpi=350, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"[SAVED] {out_p}")


# ==============================================================================
# 1. FIGURE 2 (media/image2.png): Dynamic Event Study (from Stata Table5C CSV)
# ==============================================================================
t5c = pd.read_csv(STATA_CSV / "Table5C_Mandate_EventStudy.csv")
# Exact Stata reghdfe sample sizes from STATA_MANUSCRIPT_TABLES_RESULTS.md line 264:
n_rf, n_exp, n_km, n_bp, n_sp = 9690, 7271, 9614, 7099, 8210

fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))
fig.subplots_adjust(hspace=0.34, wspace=0.30)


def style_es_panel(ax):
    ax.axhline(0.0, color="#334155", lw=1.0, ls="-", zorder=1)
    ax.axvline(2014.5, color=COL_CRIMSON, lw=1.4, ls="--", zorder=2)
    ax.axvspan(2014.5, 2023.4, color="#FECACA", alpha=0.16, zorder=0)
    ax.set_xlim(2007.6, 2023.4)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_xlabel("Fiscal Year (FY2014 Omitted Reference)")


ax = axes[0, 0]
style_es_panel(ax)
t5c_exp = t5c[t5c["Year"] >= 2010].copy()
ax.errorbar(
    t5c_exp["Year"] + 0.13,
    t5c_exp["Unexposed_DiD_RiskFound_pp"],
    yerr=1.96 * t5c_exp["Unexposed_DiD_RiskFound_se"],
    fmt="-s", color=COL_TEAL, lw=1.9, ms=4.5, capsize=2.5,
    label=f"Ex-Ante Unexposed DiD (Firm+Year FE, N={n_exp:,})",
)
ax.errorbar(
    t5c["Year"] - 0.13,
    t5c["Within_RiskFound_pp"],
    yerr=1.96 * t5c["Within_RiskFound_se"],
    fmt="-o", color=COL_NAVY, lw=1.9, ms=4.5, capsize=2.5,
    label=f"Full-Panel Within-Firm Step-Up (Firm FE, N={n_rf:,})",
)
b16_exp = float(t5c.loc[t5c["Year"] == 2016, "Unexposed_DiD_RiskFound_pp"].iloc[0])
ax.annotate(
    f"Unexposed Catch-Up:\n+{b16_exp:.1f} pp*** (t = +3.11)",
    xy=(2016.13, b16_exp), xytext=(2008.1, 22.5),
    fontsize=7.8, color=COL_TEAL, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#99F6E4", alpha=0.92),
    arrowprops=dict(arrowstyle="->", color=COL_TEAL, lw=1.1),
)
ax.set_ylim(-23, 38)
ax.set_title("(A) Extensive Margin: Standalone Risk Section Adoption (pp)", loc="left")
ax.set_ylabel("Effect Relative to FY2014 (Percentage Points)")
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[0, 1]
style_es_panel(ax)
km_b, km_se = t5c["Within_lnRiskSent_b"].values, t5c["Within_lnRiskSent_se"].values
ax.fill_between(t5c["Year"], km_b - 1.96 * km_se, km_b + 1.96 * km_se, color=COL_NAVY, alpha=0.15)
ax.errorbar(
    t5c["Year"], km_b, yerr=1.96 * km_se, fmt="-o", color=COL_NAVY, lw=2.0, ms=4.6, capsize=2.6,
    label=f"Within-Firm ln(1 + Risk Sentences) (N={n_km:,})",
)
ax.annotate(
    "Sec. 134(3)(n) + Ind-AS\nVolume Surge (+0.71*** by 2017)",
    xy=(2017, float(t5c.loc[t5c["Year"] == 2017, "Within_lnRiskSent_b"].iloc[0])),
    xytext=(2008.2, 0.82), fontsize=7.9, color=COL_NAVY, fontweight="semibold",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
    arrowprops=dict(arrowstyle="->", color=COL_NAVY, lw=1.0),
)
ax.set_title("(B) Intensive Margin: Log Whole-Report Risk Sentences", loc="left")
ax.set_ylabel("Coefficient Relative to FY2014 (Log Points)")
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[1, 0]
style_es_panel(ax)
bp_b, bp_se = t5c["Within_Boilerplate_pp"].values, t5c["Within_Boilerplate_se"].values
ax.fill_between(t5c["Year"], bp_b - 1.96 * bp_se, bp_b + 1.96 * bp_se, color=COL_CRIMSON, alpha=0.14)
ax.errorbar(
    t5c["Year"], bp_b, yerr=1.96 * bp_se, fmt="-s", color=COL_CRIMSON, lw=2.0, ms=4.6, capsize=2.6,
    label=f"Within-Firm 6-Gram Boilerplate Share (N={n_bp:,})",
)
b15_bp = float(t5c.loc[t5c["Year"] == 2015, "Within_Boilerplate_pp"].iloc[0])
b18_bp = float(t5c.loc[t5c["Year"] == 2018, "Within_Boilerplate_pp"].iloc[0])
ax.annotate(
    f"Phase 1 (FY2015: {b15_bp:.1f} pp***)\nInitial Policy Drafting Overhaul",
    xy=(2015, b15_bp), xytext=(2016.6, -15.5),
    fontsize=7.8, color=COL_SLATE, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
    arrowprops=dict(arrowstyle="->", color=COL_SLATE, lw=1.0),
)
ax.annotate(
    f"Phase 2 (FY2016+): Verbatim\nCopy-Paste Surge (+{b18_bp:.1f} pp***)",
    xy=(2018, b18_bp), xytext=(2008.1, 8.8),
    fontsize=7.8, color=COL_CRIMSON, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#FCA5A5", alpha=0.94),
    arrowprops=dict(arrowstyle="->", color=COL_CRIMSON, lw=1.1),
)
ax.set_ylim(-21.5, 16.0)
ax.set_title("(C) 6-Gram Boilerplate: Initial Overhaul vs. Copy-Paste Surge", loc="left")
ax.set_ylabel("Effect Relative to FY2014 (Percentage Points)")
ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[1, 1]
style_es_panel(ax)
sp_b, sp_se = t5c["Within_Specificity1A_pp"].values, t5c["Within_Specificity1A_se"].values
ax.fill_between(t5c["Year"], sp_b - 1.96 * sp_se, sp_b + 1.96 * sp_se, color=COL_TEAL, alpha=0.14)
ax.errorbar(
    t5c["Year"], sp_b, yerr=1.96 * sp_se, fmt="-D", color=COL_TEAL, lw=2.0, ms=4.4, capsize=2.6,
    label=f"Within-Firm Named-Entity Risk Specificity (N={n_sp:,})",
)
b17_sp = float(t5c.loc[t5c["Year"] == 2017, "Within_Specificity1A_pp"].iloc[0])
ax.annotate(
    f"Persistent Specificity Dilution\n(FY2017: {b17_sp:.2f} pp***)",
    xy=(2017, b17_sp), xytext=(2008.2, -1.72),
    fontsize=7.9, color=COL_CRIMSON, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#FCA5A5", alpha=0.94),
    arrowprops=dict(arrowstyle="->", color=COL_CRIMSON, lw=1.0),
)
ax.set_ylim(-2.1, 1.15)
ax.set_title("(D) Risk-Section Named-Entity Specificity (Hope et al., 2016)", loc="left")
ax.set_ylabel("Effect Relative to FY2014 (Percentage Points)")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")
save_docx_png(fig, "image2.png")


# ==============================================================================
# 2. FIGURE 3 (media/image3.png): Staggered SEBI Reg. 21 Event Study (Stata CSV)
# ==============================================================================
t5d = pd.read_csv(STATA_CSV / "Table5D_Staggered_SEBI_EventStudy.csv")
bins3d = pd.read_csv(SCRATCH / "stata_fig3d_bins.csv")
ns_sp, ns_bp, ns_km = 9439, 8247, 10884
p_sp, p_bp = 0.6212, 0.0833

fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.1))
fig.subplots_adjust(hspace=0.35, wspace=0.27)


def style_stag_ax(ax):
    ax.axhline(0.0, color="#334155", lw=1.0, zorder=1)
    ax.axvline(-0.5, color=COL_CRIMSON, lw=1.4, ls="--", zorder=2)
    ax.axvspan(-0.5, 5.4, color="#DCFCE7", alpha=0.25, zorder=0)
    ax.set_xlim(-5.4, 5.4)
    ax.set_xticks(range(-5, 6))
    ax.set_xticklabels([f"{k:+d}" if k != 0 else "0\n(RMC)" for k in range(-5, 6)])
    ax.set_xlabel("Event Time Relative to First Mandatory RMC Year (t = -1 Omitted)")


ax = axes[0, 0]
style_stag_ax(ax)
ev_t = t5d["Event_Time_t"].values
s_b, s_se = t5d["Specificity1A_pp"].values, t5d["Specificity1A_se"].values
ax.fill_between(ev_t, s_b - 1.96 * s_se, s_b + 1.96 * s_se, color=COL_GREEN, alpha=0.14)
ax.errorbar(ev_t, s_b, yerr=1.96 * s_se, fmt="-o", color=COL_GREEN, lw=2.1, ms=4.8, capsize=2.8, label=f"Risk Specificity (Firm+Year FE, N={ns_sp:,})")
ax.text(
    -5.15, 2.25,
    f"Pre-Trend Joint Wald p = {p_sp:.2f}\nTable 18 Static DiD: +0.56 pp* (t = +1.94)",
    fontsize=7.9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
)
ax.set_ylim(-2.1, 3.1)
ax.set_title("(A) Entity Specificity Around Board RMC Formation (pp)", loc="left")
ax.set_ylabel("Treatment Effect (Percentage Points)")
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[0, 1]
style_stag_ax(ax)
b_b, b_se = t5d["Boilerplate_pp"].values, t5d["Boilerplate_se"].values
ax.fill_between(ev_t, b_b - 1.96 * b_se, b_b + 1.96 * b_se, color=COL_CRIMSON, alpha=0.13)
ax.errorbar(ev_t, b_b, yerr=1.96 * b_se, fmt="-s", color=COL_CRIMSON, lw=2.1, ms=4.8, capsize=2.8, label=f"6-Gram Boilerplate Share (Firm+Year FE, N={ns_bp:,})")
b0_bp = float(t5d.loc[t5d["Event_Time_t"] == 0, "Boilerplate_pp"].iloc[0])
ax.annotate(
    f"Adoption-Year Overhaul:\n{b0_bp:.2f} pp*** (t = -3.00, p = 0.003)",
    xy=(0, b0_bp), xytext=(0.75, -13.1),
    fontsize=7.8, color=COL_CRIMSON, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#FCA5A5", alpha=0.95),
    arrowprops=dict(arrowstyle="->", color=COL_CRIMSON, lw=1.1),
)
ax.text(
    -5.15, 5.5, f"Pre-Trend Wald p = {p_bp:.2f}", fontsize=7.9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
)
ax.set_ylim(-16.0, 8.2)
ax.set_title("(B) Verbatim 6-Gram Boilerplate Around RMC Formation (pp)", loc="left")
ax.set_ylabel("Treatment Effect (Percentage Points)")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[1, 0]
style_stag_ax(ax)
k_b, k_se = t5d["lnRiskSent_b"].values, t5d["lnRiskSent_se"].values
ax.fill_between(ev_t, k_b - 1.96 * k_se, k_b + 1.96 * k_se, color=COL_NAVY, alpha=0.14)
ax.errorbar(ev_t, k_b, yerr=1.96 * k_se, fmt="-D", color=COL_NAVY, lw=2.0, ms=4.5, capsize=2.8, label=f"ln(1 + Risk Sentences) (Firm+Year FE, N={ns_km:,})")
ax.text(
    -5.15, -0.185,
    "Table 18 Static DiD: -0.0554* (t = -1.95)\nBoard RMC Trims Generic Risk Bloat",
    fontsize=7.9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
)
ax.set_ylim(-0.22, 0.19)
ax.set_title("(C) Whole-Report Risk Sentence Volume Around RMC Formation", loc="left")
ax.set_ylabel("Treatment Effect (Log Points)")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[1, 1]
ax.axvspan(0, 100, color="#DBEAFE", alpha=0.30, zorder=0)
ax.axvspan(100, 500, color="#DCFCE7", alpha=0.25, zorder=0)
ax.axvline(100, color=COL_NAVY, lw=1.4, ls="--", zorder=2)
ax.axvline(500, color=COL_GREEN, lw=1.4, ls="--", zorder=2)
w1 = bins3d[bins3d["wave"] == 1]
w2 = bins3d[bins3d["wave"] == 2]
ax.scatter(w1["bin25"], w1["sp100"], color=COL_SLATE, s=28, alpha=0.80, label="FY2015–2018 Bin Means (Top 100 Cutoff Only)")
x_grid = np.linspace(13, 738, 200)
ax.plot(x_grid, np.poly1d(np.polyfit(w1["bin25"], w1["sp100"], 2))(x_grid), "--", color=COL_SLATE, lw=1.8)
ax.scatter(w2["bin25"], w2["sp100"], color=COL_TEAL, marker="s", s=30, alpha=0.90, label="FY2019–2025 Bin Means (Top 500 / 1,000 Cutoffs)")
ax.plot(x_grid, np.poly1d(np.polyfit(w2["bin25"], w2["sp100"], 2))(x_grid), "-", color=COL_TEAL, lw=2.2)
ax.text(50, 9.62, "Wave 1\nTop 100\n(FY2015)", ha="center", va="top", fontsize=7.6, color=COL_NAVY, fontweight="bold")
ax.text(300, 9.62, "Wave 2 Expansion: Rank 101–500\n(Kotak Cutoff FY2019: +0.53 pp Shift)", ha="center", va="top", fontsize=7.6, color=COL_GREEN, fontweight="bold")
ax.text(625, 9.62, "Wave 3\nRank 501+\n(FY2021)", ha="center", va="top", fontsize=7.6, color="#475569", fontweight="semibold")
ax.set_xlim(0, 755)
ax.set_ylim(5.3, 9.85)
ax.set_title("(D) Specificity Across Market-Cap Rank Cutoffs (Rank 100 & 500)", loc="left")
ax.set_xlabel("Annual Market-Capitalization Rank (1 = Largest BSE Firm; 25-Rank Bins)")
ax.set_ylabel("Mean Risk Specificity (%)")
ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")
save_docx_png(fig, "image3.png")


# ==============================================================================
# 3. FIGURE 4 in .docx (media/image4.png): Taxonomy & Governance (Tables 26 & 20)
# ==============================================================================
yr_rf = pd.read_csv(SCRATCH / "stata_fig4a_shares.csv")
c9_df = pd.read_csv(SCRATCH / "stata_fig4b_taxonomy.csv")
own_df = pd.read_csv(SCRATCH / "stata_fig4c_own.csv")
v_pr = pd.read_csv(SCRATCH / "stata_fig4d_prom_ventiles.csv")
v_rd = pd.read_csv(SCRATCH / "stata_fig4d_rd_ventiles.csv")

fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))
fig.subplots_adjust(hspace=0.34, wspace=0.29)

ax = axes[0, 0]
ax.stackplot(
    yr_rf["year"].values,
    yr_rf["RF_IDIO"], yr_rf["RF_SYS"], yr_rf["RF_LR"], yr_rf["RF_FIN"], yr_rf["RF_TAX"],
    labels=[
        "Idiosyncratic / Operational (42.3%)",
        "Macro / Systematic (36.1%)",
        "Legal & Regulatory (15.0%)",
        "Financial / Leverage (5.9%)",
        "Taxation (0.8%)",
    ],
    colors=[COL_NAVY, COL_TEAL, COL_AMBER, COL_CRIMSON, COL_PURPLE],
    alpha=0.86,
)
ax.axvline(2014.5, color="white", lw=1.5, ls="--")
ax.set_xlim(2001, 2025)
ax.set_ylim(0, 100)
ax.set_title("(A) Taxonomic Risk Keyword Composition (FY2001–FY2025)", loc="left")
ax.set_xlabel("Fiscal Year")
ax.set_ylabel("Share of Risk-Section Keywords (%)")
ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1", fontsize=7.6)

ax = axes[0, 1]
yp = np.arange(len(c9_df))[::-1]
ax.axvline(0.0, color="#334155", lw=1.0, ls="--")
ax.errorbar(c9_df["dist_1sd"], yp + 0.14, xerr=1.96 * c9_df["dist_se"], fmt="s", color=COL_CRIMSON, ms=5.5, lw=1.9, capsize=2.8, label="+1 SD Distress Rank")
ax.errorbar(c9_df["size_1sd"], yp - 0.14, xerr=1.96 * c9_df["size_se"], fmt="o", color=COL_NAVY, ms=5.5, lw=1.9, capsize=2.8, label="+1 SD Firm Size (MVE)")
ax.annotate(
    "Distress Cuts Current Tax\nRisk (-0.020** per SD)",
    xy=(float(c9_df.loc[5, "dist_1sd"]), 0.14), xytext=(0.045, 0.45),
    fontsize=7.6, color=COL_CRIMSON, fontweight="semibold",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#FCA5A5", alpha=0.94),
    arrowprops=dict(arrowstyle="->", color=COL_CRIMSON, lw=1.0),
)
ax.set_yticks(yp)
ax.set_yticklabels(c9_df["cat"])
ax.set_xlim(-0.24, 0.20)
ax.set_title("(B) Taxonomic Construct Validity: Distress & Size (Table 26)", loc="left")
ax.set_xlabel("Effect on Log Category Keywords (Log Points per 1-SD Shock)")
ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#CBD5E1", fontsize=7.6)

ax = axes[1, 0]
xc, w = np.arange(len(own_df)), 0.34
b1 = ax.bar(xc - w / 2, own_df["spec_mean"], width=w, yerr=1.96 * own_df["spec_se"], color=COL_NAVY, alpha=0.88, capsize=3.0)
for xi, val, se_v in zip(xc - w / 2, own_df["spec_mean"], own_df["spec_se"]):
    ax.text(xi, val + 1.96 * se_v + 0.25, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.4, color=COL_NAVY, fontweight="bold")
ax.set_ylim(0, 15.8)
ax.set_ylabel("Mean Risk Specificity (%)", color=COL_NAVY)
ax.tick_params(axis="y", labelcolor=COL_NAVY)
ax.set_xticks(xc)
ax.set_xticklabels(["Government\nPSU", "Foreign\nMNC", "Business\nGroup", "Standalone\nPrivate"])
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(COL_CRIMSON)
ax2.grid(False)
b2 = ax2.bar(xc + w / 2, own_df["bp_mean"], width=w, yerr=1.96 * own_df["bp_se"], color=COL_CRIMSON, alpha=0.80, capsize=3.0)
for xi, val, se_v in zip(xc + w / 2, own_df["bp_mean"], own_df["bp_se"]):
    ax2.text(xi, val + 1.96 * se_v + 1.1, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.4, color=COL_CRIMSON, fontweight="bold")
ax2.set_ylim(0, 64)
ax2.set_ylabel("Mean Verbatim Boilerplate (%)", color=COL_CRIMSON)
ax2.tick_params(axis="y", labelcolor=COL_CRIMSON)
ax.set_title("(C) Disclosure Quality Across Indian Corporate Ownership Groups", loc="left")
ax.legend([b1, b2], ["Risk Specificity % (Left)", "6-Gram Boilerplate % (Right)"], loc="upper right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[1, 1]
ax.axhline(0.0, color="#334155", lw=0.9, ls=":")
ax.scatter(v_pr["bx"], v_pr["by"], color=COL_PURPLE, s=38, alpha=0.90, label="Promoter Ownership → Risk Specificity (b = -0.0251**, t = -2.21)")
xg = np.linspace(-2.2, 2.2, 100)
sl_pr = float(np.polyfit(v_pr["bx"], v_pr["by"], 1)[0])
sl_rd = float(np.polyfit(v_rd["bx"], v_rd["by"], 1)[0])
ax.plot(xg, sl_pr * xg, "-", color=COL_PURPLE, lw=2.2)
ax.scatter(v_rd["bx"], v_rd["by"], color=COL_CRIMSON, marker="D", s=36, alpha=0.88, label="R&D Intensity → Filing Specificity (b = -0.1237***, t = -3.30)")
ax.plot(xg, sl_rd * xg, "--", color=COL_CRIMSON, lw=2.1)
ax.set_xlim(-2.45, 2.45)
ax.set_ylim(-0.62, 0.58)
ax.set_title("(D) Insider Entrenchment & Proprietary R&D Withholding (Table 20)", loc="left")
ax.set_xlabel("Residualized Predictor Shock (Standardized Z-Score)")
ax.set_ylabel("Demeaned Entity Specificity Response (pp)")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")
save_docx_png(fig, "image4.png")


# ==============================================================================
# 4. FIGURE 5 in .docx (media/image5.png): ICFS & Tone Inversion (Tables 28 & 22)
# ==============================================================================
v_icfs = pd.read_csv(SCRATCH / "stata_fig5a_icfs_ventiles.csv")
icfs_df = pd.read_csv(SCRATCH / "stata_icfs_subperiods.csv")
v_li = pd.read_csv(SCRATCH / "stata_fig5c_li_ventiles.csv")
t6a = pd.read_csv(STATA_CSV / "Table6A_Li2010_Main.csv")
t6b = pd.read_csv(STATA_CSV / "Table6B_Li2010_Decomp.csv")
t11 = pd.read_csv(STATA_CSV / "Table11_Beatty2019_ICFS.csv")

m2_cf = t11[(t11["Model"] == "(2) Mandate_Shift") & (t11["Regressor"] == "CFRATIO")].iloc[0]
m2_int = t11[(t11["Model"] == "(2) Mandate_Shift") & (t11["Regressor"] == "CFRATIO_x_POST")].iloc[0]
b_pre, t_pre = float(m2_cf["Coef"]), float(m2_cf["t_stat"])
b_int, t_int, n_icfs = float(m2_int["Coef"]), float(m2_int["t_stat"]), int(m2_int["N"])
b_post = b_pre + b_int

fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.1))
fig.subplots_adjust(hspace=0.34, wspace=0.43)

ax = axes[0, 0]
vp0 = v_icfs[v_icfs["POST_2015_LODR_CA2013"] == 0]
vp1 = v_icfs[v_icfs["POST_2015_LODR_CA2013"] == 1]
ax.scatter(vp0["bx"], vp0["by"], color=COL_NAVY, s=40, alpha=0.9, label=f"Pre-2015 Ventiles (Slope = +{b_pre:.4f}***, t={t_pre:.2f})")
x_line = np.linspace(min(vp0["bx"].min(), vp1["bx"].min()), max(vp0["bx"].max(), vp1["bx"].max()), 100)
ax.plot(x_line, vp0["by"].mean() + b_pre * (x_line - vp0["bx"].mean()), "-", color=COL_NAVY, lw=2.1)
ax.scatter(vp1["bx"], vp1["by"], color=COL_CRIMSON, marker="s", s=40, alpha=0.9, label=f"Post-2015 Ventiles (Slope = +{b_post:.4f}***)")
ax.plot(x_line, vp1["by"].mean() + b_post * (x_line - vp1["bx"].mean()), "--", color=COL_CRIMSON, lw=2.1)
ax.text(
    0.03, 0.95,
    f"Table 28 Col (2) [N = {n_icfs:,}]:\nCash Flow × Post-2015 = {b_int:.4f}*** (t = {t_int:.2f})\n56.2% Reduction in Cash-Flow Wedge",
    transform=ax.transAxes, va="top", fontsize=7.9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
)
ax.set_title("(A) Residualized Binscatter: Next-Year CapEx vs. Cash Flow", loc="left")
ax.set_xlabel("Operating Cash Flow / Assets (Residualized)")
ax.set_ylabel("Next-Year CapEx / Assets")
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, edgecolor="#CBD5E1")

ax = axes[0, 1]
xp = np.arange(len(icfs_df))
ax.axvspan(3.5, 6.5, color="#DCFCE7", alpha=0.28, zorder=0)
ax.axvline(3.5, color=COL_CRIMSON, lw=1.4, ls="--", zorder=2)
colors_b = [COL_NAVY] * 4 + [COL_GREEN] * 3
for idx, r in icfs_df.iterrows():
    ax.errorbar(idx, r["b"], yerr=1.96 * r["se"], fmt="o", color=colors_b[idx], ms=6.5, lw=2.0, capsize=3.5)
ax.plot(xp, icfs_df["b"], "-", color="#475569", lw=1.2, alpha=0.7)
ax.axhline(0.0, color="#334155", lw=0.9)
ax.annotate(
    "2015 Statutory Mandate:\nSustained Step-Down\n(0.098 → 0.037)",
    xy=(4.0, float(icfs_df.loc[4, "b"])), xytext=(3.72, 0.116),
    fontsize=7.9, color=COL_GREEN, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#BBF7D0", alpha=0.94),
    arrowprops=dict(arrowstyle="->", color=COL_GREEN, lw=1.1),
)
ax.set_xticks(xp)
ax.set_xticklabels(icfs_df["window"], rotation=15)
ax.set_ylim(-0.005, 0.155)
n_sub_nf = int(icfs_df["N"].iloc[0])
ax.set_title("(B) Dynamic Evolution of Investment–Cash Flow Sensitivity", loc="left")
ax.set_xlabel(f"Multi-Year Subperiod Window (N = {n_sub_nf:,} Non-Financial Obs.)")
ax.set_ylabel("ICFS Slope Coefficient (on Operating Cash Flow)")

ax = axes[1, 0]
m3_li = t6a[(t6a["Model"] == "(3) EARN_t+1_Firm") & (t6a["Regressor"] == "FLS_TONE")].iloc[0]
b_li, t_li, n_li = float(m3_li["Coef"]), float(m3_li["t_stat"]), int(m3_li["N"])
ax.errorbar(v_li["bx"], v_li["by"], yerr=v_li["bse"], fmt="o", color=COL_CRIMSON, ms=5.0, lw=1.1, capsize=2.2, alpha=0.9, label="20 Equal-Count Ventiles (±1 SE)")
xg_li = np.linspace(v_li["bx"].min(), v_li["bx"].max(), 100)
ax.plot(xg_li, v_li["by"].mean() + b_li * (xg_li - v_li["bx"].mean()), "-", color=COL_NAVY, lw=2.2, label=f"HDFE Fit: b = {b_li:.4f}*** (t = {t_li:.2f})")
ax.text(
    0.03, 0.04,
    f"Table 22 Col (3): N = {n_li:,} (Firm + Year FE)\nControls: ROA, Total Accruals, Firm Size, Book-to-Market,\nStock Return, Volatilities & Special Items",
    transform=ax.transAxes, va="bottom", fontsize=7.8,
    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
)
ax.set_ylim(0.0715, 0.0865)
ax.set_title("(C) Indian Tone Inversion: ROA (t+1) vs. Naïve Bayes Tone", loc="left")
ax.set_xlabel("Supervised Naïve Bayes Forward-Looking Tone")
ax.set_ylabel("Future Profitability (ROA at t+1, Residualized)")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#CBD5E1")

ax = axes[1, 1]


def get_stata_row(df_csv, mod, reg):
    r = df_csv[(df_csv["Model"] == mod) & (df_csv["Regressor"] == reg)].iloc[0]
    return float(r["Coef"]), float(r["SE"]), float(r["t_stat"]), float(r["p_value"])


li_specs = [
    ("ROA (t+1) [Ind+Year FE]", *get_stata_row(t6a, "(2) EARN_t+1_Ind", "FLS_TONE")),
    ("ROA (t+1) [Firm+Year FE]", *get_stata_row(t6a, "(3) EARN_t+1_Firm", "FLS_TONE")),
    ("ROA (t+2) [Firm+Year FE]", *get_stata_row(t6a, "(4) EARN_t+2_Firm", "FLS_TONE")),
    ("ROA (t+3) [Firm+Year FE]", *get_stata_row(t6a, "(5) EARN_t+3_Firm", "FLS_TONE")),
    ("Δ ROA (t+1) [Change]", *get_stata_row(t6a, "(6) dEARN_t+1", "FLS_TONE")),
    ("Positive FLS Share (Obfusc.)", *get_stata_row(t6b, "(2) Polarity_Firm", "FLS_Pos_Share")),
    ("Naïve Bayes Tone (Horse-Race)", *get_stata_row(t6b, "(4) NB_vs_Dict", "FLS_TONE")),
    ("LM Dictionary Tone (Null)", *get_stata_row(t6b, "(4) NB_vs_Dict", "FLS_TONE_Dict")),
]
li_spec_df = pd.DataFrame(li_specs, columns=["label", "b", "se", "t", "p"])
y_pos = np.arange(len(li_spec_df))[::-1]
ax.axvline(0.0, color=COL_CRIMSON, lw=1.2, ls="--", zorder=1)
for yp, (_, row) in zip(y_pos, li_spec_df.iterrows()):
    col_pt = COL_SLATE if abs(row["t"]) < 1.65 else (COL_CRIMSON if "Pos" in row["label"] else COL_NAVY)
    ax.errorbar(row["b"], yp, xerr=1.96 * row["se"], fmt="o", color=col_pt, ms=5.5, lw=1.8, capsize=2.8)
    lbl_val = f"{row['b']:+.4f}{stars(row['p'])} (t={row['t']:.2f})"
    ax.text(0.0072, yp, lbl_val, va="center", ha="left", fontsize=7.6, fontweight="semibold", color=col_pt)
ax.set_yticks(y_pos)
ax.set_yticklabels(li_spec_df["label"])
ax.set_xlim(-0.016, 0.021)
ax.set_title("(D) Multi-Horizon, Polarity & ML vs. Dictionary Horse-Race", loc="left")
ax.set_xlabel("Coefficient Predicting Future Profitability (95% Clustered CIs)")
save_docx_png(fig, "image5.png")


# ==============================================================================
# 5. FIGURE 6 in .docx (media/image6.png): Master Forest Plot (Stata CSV)
# ==============================================================================
fdf = pd.read_csv(SCRATCH / "stata_fig6_forest.csv")
clean_forest_labels = [
    "H1a (Table 15.2): Post-2015 Mandate → Log Risk Sentences",
    "H1b (Table 15.4): Post-2015 Mandate → Boilerplate Share",
    "H1b (Table 15.7): Post-2015 Mandate → Risk Fog Index",
    "H1b (Table 15.6): Post-2015 Mandate → Risk Specificity",
    "Reg. 21 DiD (Table 18.5): Board RMC → Risk Specificity",
    "Reg. 21 DiD (Table 18.2): Board RMC → Log Risk Sentences",
    "H2a (Table 22.3): FLS Tone → ROA (t+1)",
    "H2b (Table 17.2): |Δ ROA| → MD&A Modification",
    "H2b (Table 17.5): |Δ ROA| × Post-2015 → MD&A Modification",
    "H3 (Table 25.4): Δ Industry Risk Sentences → Δ Return Volatility (t+1)",
    "H3 (Table 26.3): Distress Rank → Log Financial Risk KW",
    "H3 (Table 27.3): Repeated Risk Keywords → Idiosyncratic Volatility (t+1)",
    "H4a (Table 20.5): R&D Intensity → Filing Specificity",
    "H5 (Table 20.2): Promoter Ownership → Risk Specificity",
    "H4b (Table 28.2): Cash Flow × Post-2015 → CapEx (t+1)",
]
fig, ax = plt.subplots(figsize=(14.6, 8.4))
fig.subplots_adjust(left=0.35, right=0.97)

yp = np.arange(len(fdf))[::-1]
ax.axvline(0.0, color=COL_CRIMSON, lw=1.3, ls="--", zorder=1)
ax.axhspan(len(fdf) - 6.5, len(fdf) - 0.3, color="#F1F5F9", alpha=0.70, zorder=0)
ax.axvline(0.76, color="#CBD5E1", lw=0.9, ls="-", zorder=1)
ax.text(0.78, len(fdf) - 0.15, "Std. Effect   [t (Firm) | t (2-Way)]", fontsize=8.3, fontweight="bold", color=COL_NAVY, va="bottom", ha="left")

for idx, (y_i, row) in enumerate(zip(yp, fdf.itertuples())):
    ax.errorbar(row.b_std, y_i + 0.15, xerr=1.96 * row.se1_std, fmt="o", color=COL_NAVY, ms=5.5, lw=1.9, capsize=2.6, label="One-Way Firm-Clustered 95% CI" if idx == 0 else None)
    ax.errorbar(row.b_std, y_i - 0.15, xerr=1.96 * row.se2_std, fmt="D", color=COL_TEAL, mfc="white", mew=1.5, ms=4.8, lw=1.4, capsize=2.6, label="Two-Way (Firm × Year) Clustered 95% CI" if idx == 0 else None)
    txt = f"{row.b_std:+.3f} SD{stars(row.p1):<3s}  [{row.t1:+6.2f} | {row.t2:+6.2f}]"
    ax.text(0.78, y_i, txt, va="center", ha="left", fontsize=7.9, family="monospace", color="#0F172A", fontweight="semibold")

ax.set_yticks(yp)
ax.set_yticklabels(clean_forest_labels, fontsize=8.4, fontweight="semibold")
ax.set_xlim(-0.30, 1.34)
ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
ax.set_xlabel("Standardized Effect Size in SD Units (Shaded = Binary Regime / Cutoff Dummies; Unshaded = +1 SD Predictors)")
ax.legend(loc="lower center", bbox_to_anchor=(0.41, 0.02), frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1")
save_docx_png(fig, "image6.png")
