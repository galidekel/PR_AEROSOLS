import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# ── colour palette ──────────────────────────────────────────────────────────
C_ERA5    = "#2171b5"
C_CAMS    = "#d94801"
C_ENCODE  = "#41ab5d"
C_FUSE    = "#7b2d8b"
C_HEAD    = "#08306b"
C_OUT     = "#cb181d"
C_PATCH   = "#006d2c"
C_BG      = "#f8f9fb"

def box(ax, x, y, w, h, label, sublabel=None, color="#333333", fontsize=9,
        facecolor="white", alpha=1.0, style="round,pad=0.15", lw=1.5):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle=style, linewidth=lw,
        edgecolor=color, facecolor=facecolor, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2 + (0.12 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.18, sublabel, ha="center", va="center",
                fontsize=7.5, color="#555555", zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=5)

# ════════════════════════════════════════════════════════════════════════════
# ROW 0 — section labels
# ════════════════════════════════════════════════════════════════════════════
for label, x in [("INPUTS", 3.5), ("ENCODERS", 8.2), ("FUSION", 11.8), ("OUTPUT", 14.4)]:
    ax.text(x, 9.65, label, ha="center", va="center", fontsize=9,
            color="#aaaaaa", fontweight="bold")

# ════════════════════════════════════════════════════════════════════════════
# ERA5 branch  (top)
# ════════════════════════════════════════════════════════════════════════════
# Input tensor
box(ax, 0.3, 7.2, 2.8, 1.4,
    "ERA5 Route  [t → t+7]",
    "T=28 × C=48 × 80×240\n(6-hourly, 0.5°, 7 days)",
    color=C_ERA5, facecolor="#dbeeff", fontsize=8.5)

ax.text(1.7, 7.05, "40 PL  (8 vars × 5 levels)  +  8 SL", ha="center",
        fontsize=7, color=C_ERA5)

# PatchEmbed3D
box(ax, 3.9, 7.35, 2.4, 1.1,
    "PatchEmbed3D",
    "per channel  (1×20×20)\n→ 28×4×12 = 1344 tokens",
    color=C_PATCH, facecolor="#e8f5e9", fontsize=8)

arrow(ax, 3.1, 7.95, 3.9, 7.95, color=C_ERA5)

# CrossAttentionGate (channel ← patches)
box(ax, 6.6, 7.35, 2.5, 1.1,
    "CrossAttentionGate ×48",
    "channel token ← patches\n→ [48 × D=256]",
    color=C_ENCODE, facecolor="#e8f5e9", fontsize=8)

arrow(ax, 6.3, 7.95, 6.6, 7.95, color=C_PATCH)

# ════════════════════════════════════════════════════════════════════════════
# CAMS branch  (bottom)
# ════════════════════════════════════════════════════════════════════════════
box(ax, 0.3, 4.8, 2.8, 1.4,
    "CAMS Africa Dust  [t]",
    "T=4 × 2 × 61×67\n(6-hourly, 0.75°, 1 day)",
    color=C_CAMS, facecolor="#fde8d8", fontsize=8.5)

ax.text(1.7, 4.65, "aod550  +  duaod550", ha="center",
        fontsize=7, color=C_CAMS)

# Conv3d stack
box(ax, 3.9, 4.95, 2.4, 1.1,
    "Conv3d Stack",
    "3 × (Conv3d→GELU→BN)\n32→64→128 filters",
    color=C_CAMS, facecolor="#fde8d8", fontsize=8)

arrow(ax, 3.1, 5.55, 3.9, 5.55, color=C_CAMS)

# AdaptiveAvgPool + projection
box(ax, 6.6, 4.95, 2.5, 1.1,
    "AvgPool3d + Linear",
    "AdaptiveAvgPool(1,1,1)\n→ [D=256]",
    color=C_CAMS, facecolor="#fde8d8", fontsize=8)

arrow(ax, 6.3, 5.55, 6.6, 5.55, color=C_CAMS)

# ════════════════════════════════════════════════════════════════════════════
# FUSION
# ════════════════════════════════════════════════════════════════════════════
# concat box
box(ax, 9.4, 5.9, 1.8, 1.8,
    "Concat",
    "[49 × D]\ndust + route",
    color=C_FUSE, facecolor="#f3e8ff", fontsize=8)

arrow(ax, 9.1, 7.95, 9.75, 7.7,  color=C_ENCODE, lw=1.5)
arrow(ax, 9.1, 5.55, 9.75, 6.05, color=C_CAMS,   lw=1.5)

# PR token cross-attention
box(ax, 11.5, 5.9, 2.5, 1.8,
    "CrossAttentionGate",
    "learnable PR token\n← [49 × D] summaries\n→ [D=256]",
    color=C_FUSE, facecolor="#f3e8ff", fontsize=8)

arrow(ax, 11.2, 6.8, 11.5, 6.8, color=C_FUSE)

# ════════════════════════════════════════════════════════════════════════════
# MLP HEAD
# ════════════════════════════════════════════════════════════════════════════
box(ax, 14.1, 6.15, 1.6, 1.3,
    "MLP Head",
    "LN→Linear→\nGELU→Linear",
    color=C_HEAD, facecolor="#e8eaf6", fontsize=8)

arrow(ax, 14.0, 6.8, 14.1, 6.8, color=C_FUSE)

# Output
box(ax, 14.1, 4.9, 1.6, 0.9,
    "AE 440–870",
    "scalar  @ t+7",
    color=C_OUT, facecolor="#fff0f0", fontsize=8)

arrow(ax, 14.9, 6.15, 14.9, 5.8, color=C_HEAD)

# ════════════════════════════════════════════════════════════════════════════
# Positional embedding note
# ════════════════════════════════════════════════════════════════════════════
ax.text(5.1, 8.6, "+ pos embed", ha="center", fontsize=7,
        color=C_PATCH, style="italic")

# ════════════════════════════════════════════════════════════════════════════
# Target label
# ════════════════════════════════════════════════════════════════════════════
ax.text(14.9, 4.65, "AERONET  Cape San Juan\nt+7  (regression)",
        ha="center", fontsize=7, color=C_OUT)

# ════════════════════════════════════════════════════════════════════════════
# Dimension flow annotations
# ════════════════════════════════════════════════════════════════════════════
for txt, x, y, col in [
    ("[B,1,T,H,W]\nper ch.", 5.25, 7.25, C_PATCH),
    ("[B,C_r,D]",            9.25, 8.2,  C_ENCODE),
    ("[B,D]",                9.25, 5.3,  C_CAMS),
    ("[B,D]",                13.95, 6.95, C_FUSE),
]:
    ax.text(x, y, txt, ha="center", fontsize=7, color=col, style="italic")

# ════════════════════════════════════════════════════════════════════════════
# Title
# ════════════════════════════════════════════════════════════════════════════
ax.set_title("PRAfricaDustRoute — Model Architecture",
             fontsize=13, fontweight="bold", pad=8)

plt.tight_layout()
plt.savefig("architecture_schema.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved architecture_schema.png")
plt.show()
