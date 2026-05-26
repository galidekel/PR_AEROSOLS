import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# ── data domain ────────────────────────────────────────────────────────────
N, W, S, E = 35.0, -90.0, -10.0, 30.0
PR_LON, PR_LAT = -65.62, 18.38

# ── colours ────────────────────────────────────────────────────────────────
C_OCEAN    = "#cfe8f3"
C_LAND     = "#f2e5c4"
C_BOX      = "#2171b5"
C_AFRICA   = "#d94801"
C_PR       = "red"

# ── figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# bigger background — 20° padding on each side
ax.set_extent([W - 20, E + 20, S - 15, N + 15], crs=ccrs.PlateCarree())

# ── basemap ────────────────────────────────────────────────────────────────
ax.add_feature(cfeature.OCEAN,     facecolor=C_OCEAN, zorder=0)
ax.add_feature(cfeature.LAND,      facecolor=C_LAND,  zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6,     zorder=3)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.5, zorder=3)

# ── data domain rectangle ──────────────────────────────────────────────────
ax.add_patch(mpatches.Rectangle(
    (W, S), width=E - W, height=N - S,
    linewidth=0, facecolor=C_BOX, alpha=0.20,
    transform=ccrs.PlateCarree(), zorder=2
))
ax.plot(
    [W, E, E, W, W],
    [S, S, N, N, S],
    color=C_BOX, linewidth=2.5,
    transform=ccrs.PlateCarree(), zorder=4
)

# ── CAMS Africa domain ─────────────────────────────────────────────────────
AN, AW, AS, AE = 35.0, -17.0, -10.0, 30.0
ax.add_patch(mpatches.Rectangle(
    (AW, AS), width=AE - AW, height=AN - AS,
    linewidth=0, facecolor=C_AFRICA, alpha=0.20,
    transform=ccrs.PlateCarree(), zorder=3
))
ax.plot(
    [AW, AE, AE, AW, AW],
    [AS, AS, AN, AN, AS],
    color=C_AFRICA, linewidth=2.5, linestyle="--",
    transform=ccrs.PlateCarree(), zorder=5
)
ax.text(15.0, 12.0, "Africa",
        ha="center", va="center", fontsize=13, color=C_AFRICA,
        fontweight="bold", transform=ccrs.PlateCarree(), zorder=11,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

PAD = 1.5   # degrees offset so labels sit just outside the corners
corners = [
    (W - PAD, N + PAD, "right", "bottom", f"90°W, 35°N", C_BOX),
    (E + PAD, N + PAD, "left",  "bottom", f"30°E, 35°N", C_BOX),
    (W - PAD, S - PAD, "right", "top",    f"90°W, 10°S", C_BOX),
    (E + PAD, S - PAD, "left",  "top",    f"30°E, 10°S", C_BOX),
]
for lon, lat, ha, va, label, color in corners:
    ax.text(lon, lat, label,
            ha=ha, va=va, fontsize=8.5, color=color, fontweight="bold",
            transform=ccrs.PlateCarree(), zorder=11,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# ── ERA5 grid (0.25°) over full domain ────────────────────────────────────
def draw_grid(ax, lon0, lon1, lat0, lat1, res, color, lw, alpha, zorder,
             excl=None):
    """Draw a regular grid; excl=(xlo,xhi,ylo,yhi) masks out a rectangle."""
    segs_x, segs_y = [], []
    # vertical lines
    for lon in np.arange(lon0, lon1 + res * 0.5, res):
        if excl and excl[0] <= lon <= excl[1]:
            # split into below and above exclusion zone
            for seg in [(lat0, excl[2]), (excl[3], lat1)]:
                if seg[1] > seg[0]:
                    segs_x += [lon, lon, np.nan]
                    segs_y += [seg[0], seg[1], np.nan]
        else:
            segs_x += [lon, lon, np.nan]
            segs_y += [lat0, lat1, np.nan]
    # horizontal lines
    for lat in np.arange(lat0, lat1 + res * 0.5, res):
        if excl and excl[2] <= lat <= excl[3]:
            # split into left and right of exclusion zone
            for seg in [(lon0, excl[0]), (excl[1], lon1)]:
                if seg[1] > seg[0]:
                    segs_x += [seg[0], seg[1], np.nan]
                    segs_y += [lat, lat, np.nan]
        else:
            segs_x += [lon0, lon1, np.nan]
            segs_y += [lat, lat, np.nan]
    ax.plot(segs_x, segs_y, color=color, linewidth=lw, alpha=alpha,
            transform=ccrs.PlateCarree(), zorder=zorder, rasterized=True)

# ERA5 0.25° — full domain minus CAMS Africa rectangle
draw_grid(ax, W, E, S, N, 0.25, C_BOX, 0.4, 0.30, 3,
          excl=(AW, AE, AS, AN))
# CAMS 0.75° — Africa rectangle only
draw_grid(ax, AW, AE, AS, AN, 0.75, "#a63603", 0.8, 0.55, 4)

# ── Cape San Juan station ──────────────────────────────────────────────────
ax.scatter(PR_LON, PR_LAT, s=200, color=C_PR, alpha=0.25,
           transform=ccrs.PlateCarree(), zorder=8)
ax.scatter(PR_LON, PR_LAT, s=60,  color=C_PR,
           edgecolors="darkred", linewidths=0.8,
           transform=ccrs.PlateCarree(), zorder=9)
ax.text(PR_LON + 2, PR_LAT + 1.5, "Cape San Juan\n(AERONET)",
        fontsize=8.5, color="darkred", weight="bold",
        transform=ccrs.PlateCarree(), zorder=10,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# ── legend ─────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=C_BOX, edgecolor=C_BOX, linewidth=2.0,
                   label="ERA5 domain  (35°N → 10°S, 90°W → 30°E)  0.25° grid"),
    plt.Line2D([0], [0], color=C_AFRICA, linewidth=2.5, linestyle="--",
               label="CAMS Africa  (35°N → 10°S, 17°W → 30°E)  0.75° grid"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=C_PR, markeredgecolor="darkred",
               markersize=9, label="Cape San Juan AERONET"),
]
ax.legend(handles=handles, loc="lower left",
          fontsize=9, framealpha=0.92, edgecolor="#aaaaaa",
          handlelength=1.8, handletextpad=0.5)

# ── gridlines ──────────────────────────────────────────────────────────────
gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                  color="gray", alpha=0.4, linestyle=":")
gl.top_labels   = False
gl.right_labels = False
gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}

ax.set_title(
    "PRAfricaDustRoute — Data Domain",
    fontsize=13, fontweight="bold", pad=10
)

plt.tight_layout()
plt.savefig("patch_domain_map.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved patch_domain_map.png")
plt.show()
