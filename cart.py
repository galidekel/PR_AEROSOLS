import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------------------
# Create figure
# -------------------------------
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# -------------------------------
# Set region (Africa → Caribbean)
# -------------------------------
ax.set_extent([-90, 30, -10, 35])

# -------------------------------
# Background styling
# -------------------------------
ax.add_feature(cfeature.OCEAN, facecolor='#cfe8f3')   # light blue ocean
ax.add_feature(cfeature.LAND, facecolor='#f2e5c4')    # sandy land
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

# -------------------------------
# Puerto Rico base location
# -------------------------------
# -------------------------------
# Puerto Rico (reference point)
# -------------------------------
pr_lon, pr_lat = -66.1, 18.4

# -------------------------------
# Spread START points (Africa, clearly on land)
# -------------------------------
sources = [
    (-15, 13),   # West Africa (inland)
    (-10, 17),   # Western Sahara (inland)
    (-2, 20),    # Central Sahara (west)
    (6, 23),     # Central Sahara (east)
    (12, 27),    # North Africa
]

# -------------------------------
# Endpoints slightly BEFORE PR
# (so arrows don’t pile up)
# -------------------------------
targets = [
    (-65, 17.0),
    (-65, 18.5),
    (-65, 19.5),
    (-65, 20.0),
    (-65, 21),
]

# -------------------------------
# Curvatures (for natural flow)
# -------------------------------
curvatures = [-0.35, -0.2, -0.1, 0.15, 0.3]

# -------------------------------
# Arrow styling
# -------------------------------
arrow_kwargs = dict(
    arrowstyle='->',
    color='darkorange',
    lw=3,
linestyle = '--')

# -------------------------------
# Draw arrows
# -------------------------------
for (sx, sy), (tx, ty), rad in zip(sources, targets, curvatures):
    ax.annotate(
        '',
        xy=(tx, ty), xytext=(sx, sy),
        arrowprops=dict(
            **arrow_kwargs,
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.6
        ),
        transform=ccrs.PlateCarree()
    )

# Halo (for visibility)
ax.scatter(pr_lon, pr_lat,
           s=200,
           color='red',
           alpha=0.25,
           transform=ccrs.PlateCarree(),
           zorder=9)

# # Main point
# ax.scatter(pr_lon, pr_lat,
#            s=80,
#            color='red',
#            edgecolor='black',
#            linewidth=1,
#            transform=ccrs.PlateCarree(),
#            zorder=10)

# -------------------------------
# Labels (draw last)
# -------------------------------
ax.text(pr_lon + 3, pr_lat + 1.5,
        'Puerto Rico ',
        fontsize=10,
        color='darkred',
        weight='bold',
        transform=ccrs.PlateCarree(),
        zorder=11)

ax.text(-20, 25,
        'Sahara Dust Source',
        fontsize=10,
        color='saddlebrown',
        weight='bold',
        transform=ccrs.PlateCarree(),
        zorder=11)

# -------------------------------
# Title
# -------------------------------
plt.title("Transatlantic African Dust Transport",
          fontsize=14, weight='bold')

# -------------------------------
# Save high-quality image
# -------------------------------
plt.savefig("dust_transport_map_final.png",
            dpi=300,
            bbox_inches='tight')
gl = ax.gridlines(draw_labels=True,
                  linewidth=0.5,
                  color='gray',
                  alpha=0.5,
                  linestyle='--')

gl.top_labels = False
gl.right_labels = False

gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}
# -------------------------------
# Show
# -------------------------------
plt.show()