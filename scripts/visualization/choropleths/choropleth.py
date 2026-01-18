import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import geopandas as gpd
import matplotlib.ticker as mticker
from esda.moran import Moran
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, LogFormatter, ScalarFormatter
from matplotlib.colors import Normalize, LogNorm

df_inner_county = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_outer_county.csv')
df_cbsa = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_cbsa.csv')
df_msa = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_msa.csv')
df_musa = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_musa.csv')
gdf_county = gpd.read_file('F:\\dsl_CLIMA\\submittable\\source\\shape files\\county\\tl_2021_us_county.shp')
gdf_cbsa = gpd.read_file('F:\\dsl_CLIMA\\submittable\\source\\shape files\\cbsa\\tl_2021_us_cbsa.shp')

df_inner_county['user_loc'] = df_inner_county['user_loc'].astype(str).str.zfill(5)
df_cbsa['CBSA Code'] = df_cbsa['CBSA Code'].astype(str).str.zfill(5)

gdf_county['user_loc'] = (gdf_county['STATEFP'].str.zfill(2)+ gdf_county['COUNTYFP'].str.zfill(3))
gdf_cbsa.rename(columns={'GEOID': 'CBSA Code'}, inplace=True)
gdf_cbsa['CBSA Code'] = gdf_cbsa['CBSA Code'].astype(str).str.zfill(5)

gdf_county = gdf_county.merge(df_inner_county, on='user_loc', how='left')
gdf_county = gpd.GeoDataFrame(gdf_county, geometry='geometry', crs='EPSG:4326')

gdf_cbsa = gdf_cbsa.merge(df_cbsa, on='CBSA Code', how='left')
gdf_cbsa = gpd.GeoDataFrame(gdf_cbsa, geometry='geometry', crs='EPSG:4326')

def plot_choropleth_per_subplot(
    gdf,
    value_cols,
    titles=None,
    log_scale_cols=None,        # Data log-transform
    force_log_cols=None,        # Force visual log colorbar on linear data
    cmap="viridis",
    missing_color="lightgrey",
    edgecolor="white",
    linewidth=0.2,
    figsize_per_plot=5,
    continental_only=True,
    zoom_padding=0.05,
    cbar_height=0.025,
    cbar_pad=0.012,
    cbar_width_ratio=0.75,      # width ratio of colorbar relative to subplot
    linear_ticks=6,
    log_ticks_per_decade=1
):
    if titles is None:
        titles = value_cols
    if log_scale_cols is None:
        log_scale_cols = []
    if force_log_cols is None:
        force_log_cols = []

    gdf_plot = gdf.copy()

    # ---- Continental US filter ----
    if continental_only:
        continental_fips = [str(i).zfill(2) for i in range(1, 57)
                            if i not in [2, 15, 60, 66, 69, 72, 78]]
        if "STATEFP" in gdf_plot.columns:
            gdf_plot = gdf_plot[gdf_plot["STATEFP"].isin(continental_fips)]
        elif "GEOID" in gdf_plot.columns:
            gdf_plot = gdf_plot[gdf_plot["GEOID"].str[:2].isin(continental_fips)]
        else:
            raise ValueError("Cannot filter continental US: no STATEFP or GEOID column found.")

    n = len(value_cols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize_per_plot, rows * figsize_per_plot),
        constrained_layout=False
    )
    axes = axes.flatten()

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=0.08,
        hspace=0.10,
        wspace=0.05
    )

    # ---- Zoom bounds with padding ----
    minx, miny, maxx, maxy = gdf_plot.total_bounds
    dx = (maxx - minx) * zoom_padding
    dy = (maxy - miny) * zoom_padding
    bounds = (minx + dx, miny + dy, maxx - dx, maxy - dy)

    for ax, col, title in zip(axes, value_cols, titles):

        data = gdf_plot[col].replace(0, np.nan)
        vmin, vmax = data.min(), data.max()

        # ---- Determine data for plotting ----
        if col in log_scale_cols:
            # log-transform data for plotting
            plot_data = np.where(data > 0, np.log10(data), np.nan)
            norm = Normalize(vmin=np.nanmin(plot_data), vmax=np.nanmax(plot_data))
            is_log = True
        else:
            plot_data = data
            norm = Normalize(vmin=vmin, vmax=vmax)
            is_log = col in force_log_cols  # for colorbar only

        # ---- Plot
        gdf_plot.plot(
            column=plot_data,
            ax=ax,
            cmap=cmap,
            norm=norm,
            missing_kwds={"color": missing_color},
            edgecolor=edgecolor,
            linewidth=linewidth
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.axis("off")

        # ---- Colorbar below subplot ----
        pos = ax.get_position()
        width = pos.width * cbar_width_ratio
        x0 = pos.x0 + (pos.width - width) / 2
        cax = fig.add_axes([x0, pos.y0 - cbar_pad - cbar_height, width, cbar_height])

        # Create colormap for colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.get_offset_text().set_visible(False)
        cbar.ax.set_xlabel("")
        cbar.ax.tick_params(labelsize=9)

        # ---- Tick formatting ----
        if col in log_scale_cols:
            # Data is log-transformed, but show original values
            log_ticks = np.logspace(np.floor(np.log10(vmin)),
                                    np.ceil(np.log10(vmax)),
                                    int((np.ceil(np.log10(vmax)) - np.floor(np.log10(vmin))) * log_ticks_per_decade + 1))
            log_ticks = log_ticks[(log_ticks >= vmin) & (log_ticks <= vmax)]
            labels = [f"{int(t):,}" for t in log_ticks]
            cbar.ax.xaxis.set_major_locator(FixedLocator(np.log10(log_ticks)))
            cbar.ax.xaxis.set_major_formatter(FixedFormatter(labels))

        elif col in force_log_cols:
            # Linear data, forced log colorbar
            log_ticks = np.logspace(np.floor(np.log10(vmin)),
                                    np.ceil(np.log10(vmax)),
                                    int((np.ceil(np.log10(vmax)) - np.floor(np.log10(vmin))) * log_ticks_per_decade + 1))
            log_ticks = log_ticks[(log_ticks >= vmin) & (log_ticks <= vmax)]
            # Map ticks to normalized 0-1 for colorbar
            log_norm_ticks = (np.log10(log_ticks) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
            labels = [f"{int(t):,}" for t in log_ticks]
            cbar.ax.xaxis.set_major_locator(FixedLocator(log_norm_ticks * (cbar.ax.get_xlim()[1] - cbar.ax.get_xlim()[0]) + cbar.ax.get_xlim()[0]))
            cbar.ax.xaxis.set_major_formatter(FixedFormatter(labels))

        else:
            # Linear data
            ticks = np.linspace(vmin, vmax, linear_ticks)
            labels = [f"{t:.2f}" if t < 10 else f"{int(t)}" for t in ticks]
            cbar.ax.xaxis.set_major_locator(FixedLocator(ticks))
            cbar.ax.xaxis.set_major_formatter(FixedFormatter(labels))

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    return fig, axes

# --- County Choropleths ---
fig, axes = plot_choropleth_per_subplot(
    gdf=gdf_county,
    value_cols=[
        "inter_county_connections",
        "outer_county_connections",
        "total connections"
    ],
    titles=[
        "Inter-County Connections",
        "Outgoing County Connections",
        "Total County Connections"
    ],
    log_scale_cols=['inter_county_connections', 'outer_county_connections', 'total connections'],
    cmap="viridis",
    continental_only=True,
    zoom_padding=-0.075,
    log_ticks_per_decade=0.5,
    cbar_height=0.01,
    cbar_pad=0.012
)

plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\choropleths\\county_connections_choropleth.png", dpi=800, bbox_inches='tight')

fig, axes = plot_choropleth_per_subplot(
    gdf=gdf_county,
    value_cols=[
        "user_est",
        "pop_est",
        "coverage est"
    ],
    titles=[
        "County User Count Estimates",
        "County Population Estimates",
        "County Coverage Estimates"
    ],
    log_scale_cols=['user_est', 'pop_est'],
    force_log_cols=['user_est','pop_est'],
    cmap="viridis",
    continental_only=True,
    zoom_padding=-0.075,
    log_ticks_per_decade=0.5,
    cbar_height=0.01,
    cbar_pad=0.012,
    linear_ticks=5
)

plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\choropleths\\county_popstats_choropleth.png", dpi=800, bbox_inches='tight')

gdf_county['inter_county_connections per capita'] = gdf_county['inter_county_connections']/gdf_county['pop_est']
gdf_county['inter_county_connections per user'] = gdf_county['inter_county_connections']/gdf_county['user_est']

gdf_county['outer_county_connections per capita'] = gdf_county['outer_county_connections']/gdf_county['pop_est']
gdf_county['outer_county_connections per user'] = gdf_county['outer_county_connections']/gdf_county['user_est']

gdf_county['total connections per capita'] = gdf_county['total connections']/gdf_county['pop_est']
gdf_county['total connections per user'] = gdf_county['total connections']/gdf_county['user_est']


# --- Per User and Per Capita Choropleths ---
fig, axes = plot_choropleth_per_subplot(
    gdf=gdf_county,
    value_cols=[
        "inter_county_connections per capita",
        "outer_county_connections per capita",
        "total connections per capita"
    ],
    titles=[
        "Inter-County Connections per Capita",
        "Outgoing County Connections per Capita",
        "Total County Connections per Capita"
    ],
    cmap="viridis",
    continental_only=True,
    zoom_padding=-0.075,
    log_ticks_per_decade=0.5,
    cbar_height=0.01,
    cbar_pad=0.012,
    linear_ticks=5
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\choropleths\\county_percapita_choropleth.png", dpi=800, bbox_inches='tight')


fig, axes = plot_choropleth_per_subplot(
    gdf=gdf_county,
    value_cols=[
        "inter_county_connections per user",
        "outer_county_connections per user",
        "total connections per user"
    ],
    titles=[
        "Inner-County Connections per User",
        "Outgoing County Connections per User",
        "Total County Connections per User"
    ],
    cmap="viridis",
    continental_only=True,
    zoom_padding=-0.075,
    log_ticks_per_decade=0.5,
    cbar_height=0.01,
    cbar_pad=0.012,
    linear_ticks=5
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\choropleths\\county_peruser_choropleth.png", dpi=800, bbox_inches='tight')