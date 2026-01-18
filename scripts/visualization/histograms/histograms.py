import pandas as pd 
import numpy as np 
import math
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import lognorm, skewnorm, genpareto, norm, gaussian_kde

df_inner_county = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_outer_county.csv')
df_cbsa = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_cbsa.csv')
df_msa = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_msa.csv')
df_musa = pd.read_csv('F:\\dsl_CLIMA\\submittable\\export\\df_musa.csv')
df_sci = pd.read_table('F:\\dsl_CLIMA\\submittable\\source\\sci\\county_county.tsv', dtype={'user_loc':str, 'fr_loc':str, 'scaled_sci':int})

# Helper function to arrange subplots in a rectangular format
def near_square_grid(k, min_cols=3):
    cols = max(min_cols, math.ceil(np.sqrt(k)))
    rows = math.ceil(k / cols)
    return rows, cols

def plot_histograms_with_pdfs_kde(
    dfs, cols, titles=None, x_labels=None, log_x=False, no_log_cols=None,
    figsize_per_plot=4, bar_padding=0.03,
    title_fontsize=12, stats_fontsize=10, legend_fontsize=9,
    box_alpha=0.15,
    bin_max=None,
    kde_max=None
):
    if titles is None:
        titles = cols
    if x_labels is None:
        x_labels = cols

    k = len(cols)
    rows, cols_n = near_square_grid(k, min_cols=3)

    fig, axes = plt.subplots(
        rows, cols_n,
        figsize=(cols_n * figsize_per_plot * 1.3, rows * figsize_per_plot * 1.15),
        squeeze=False,
        constrained_layout=True
    )
    axes = axes.flatten()

    for idx, (ax, df_sub, col, title, xlabel) in enumerate(
        zip(axes, dfs, cols, titles, x_labels)
    ):

        # ---------- Data handling ----------
        raw_data = df_sub[col].dropna().values

        use_log = log_x
        if no_log_cols is not None:
            if col in no_log_cols or idx in no_log_cols:
                use_log = False

        if use_log:
            raw_data = raw_data[raw_data > 0]
            data = np.log10(raw_data)
        else:
            data = raw_data

        if len(data) < 5:
            ax.axis('off')
            continue

        mean_val = np.mean(data)
        median_val = np.median(data)
        variance_val = np.var(data) 
        q25, q75 = np.percentile(data, [25, 75])

        # ---------- Freedman–Diaconis bins (capped) ----------
        h = 2 * (q75 - q25) / len(data) ** (1/3)
        if h <= 0 or not np.isfinite(h):
            bins = 'auto'
        else:
            bins = int(np.ceil((data.max() - data.min()) / h))

        if bin_max is not None:
            bins = min(bins, bin_max)

        counts, bin_edges, patches = ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.6,
        color='skyblue',
        edgecolor='black'
        )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # ---------- Percentile shading ----------
        for i, patch in enumerate(patches):
            if bin_centers[i] <= q25:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.4)
            elif bin_centers[i] >= q75:
                patch.set_facecolor('plum')
                patch.set_alpha(0.4)

        # ---------- KDE (subsampled) ----------
        if kde_max is not None and len(data) > kde_max:
            kde_data = np.random.choice(data, kde_max, replace=False)
        else:
            kde_data = data


        kde = gaussian_kde(kde_data)
        x_kde = np.linspace(data.min(), data.max(), 500)
        y_kde = kde(x_kde)
        ax.plot(x_kde, y_kde, color='black', lw=2, label='KDE')

        # ---------- Parametric fits ----------
        fitted_pdfs = {}

        mu, sigma = norm.fit(data)
        fitted_pdfs['Normal'] = norm.pdf(bin_centers, mu, sigma)

        a, loc_sn, scale_sn = skewnorm.fit(data)
        fitted_pdfs['SkewNormal'] = skewnorm.pdf(bin_centers, a, loc_sn, scale_sn)

        if use_log:
            shape, loc, scale = lognorm.fit(raw_data, floc=0)
            fitted_pdfs['LogNormal'] = (
                lognorm.pdf(10**bin_centers, shape, loc, scale)
                * np.log(10)
                * 10**bin_centers
            )

        # ---------- Tail detection & GenPareto ----------
        tail_q = 25
        u = np.percentile(raw_data, tail_q)
        tail_raw = raw_data[raw_data > u] - u

        u_plot = np.log10(u) if use_log else u
        dist_left = u_plot - data.min()
        dist_right = data.max() - u_plot
        tail_side = 'right' if dist_right < dist_left else 'left'

        if len(tail_raw) > 10:
            c, loc_gp, scale_gp = genpareto.fit(tail_raw, floc=0)

            tail_mask = (10**bin_centers > u) if use_log else (bin_centers > u)
            x_tail = (
                10**bin_centers[tail_mask] - u
                if use_log else
                bin_centers[tail_mask] - u
            )

            pdf_gp = (
                genpareto.pdf(x_tail, c, loc_gp, scale_gp)
                * (np.log(10) * 10**bin_centers[tail_mask] if use_log else 1.0)
            )

            gp_full = np.zeros_like(bin_centers)
            gp_full[tail_mask] = pdf_gp
            fitted_pdfs['GenPareto (25% Tail)'] = gp_full

        # ---------- Plot PDFs ----------
        for name, pdf in fitted_pdfs.items():
            ax.plot(bin_centers, pdf, lw=2, label=name)

        # ---------- RMSE vs KDE ----------
        kde_vals = kde(bin_centers)
        metrics = {
            name: np.sqrt(np.mean((pdf - kde_vals) ** 2))
            for name, pdf in fitted_pdfs.items()
        }

        stats_lines = ["RMSE of KDE vs PDF Estimates:"]
        for name, rmse in sorted(metrics.items(), key=lambda x: x[1]):
            stats_lines.append(f"{name}: {rmse:.4f}")

        stats_lines.append(f"\nMean: {mean_val:.4f}")
        stats_lines.append(f"Median: {median_val:.4f}")
        stats_lines.append(f"Variance: {variance_val:.4f}")

        ax.scatter(mean_val, kde(mean_val), color='red', s=80, zorder=5, label='Mean')
        ax.scatter(median_val, kde(median_val), color='darkgreen', s=80, zorder=5, label='Median')

        # ---------- Legend & stats ----------
        if tail_side == 'right':
            stats_x, stats_ha = 0.02, 'left'
            legend_loc = 'upper left'
        else:
            stats_x, stats_ha = 0.98, 'right'
            legend_loc = 'upper right'

        hist_proxy = Patch(
            facecolor='skyblue',
            edgecolor='black',
            alpha=0.6,
            label='Histogram (25–75%)'
        )

        # Get existing handles FIRST
        handles, labels = ax.get_legend_handles_labels()

        # Now construct legend entries explicitly
        handles = (
            [hist_proxy]
            + handles
            + [
                Patch(facecolor='lightcoral', edgecolor='black', label='≤25th percentile'),
                Patch(facecolor='plum', edgecolor='black', label='≥75th percentile')
            ]
        )

        legend = ax.legend(
            handles=handles,
            fontsize=legend_fontsize,
            loc=legend_loc,
            frameon=True
        )

        fig = ax.figure
        fig.canvas.draw()
        legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
        stats_y = legend_bbox.y0 - 0.04

        ax.text(
            stats_x,
            stats_y,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            fontsize=stats_fontsize,
            va='top',
            ha=stats_ha,
            bbox=dict(facecolor='white', alpha=box_alpha, edgecolor='black')
        )

        # ---------- Labels ----------
        ax.set_title(title, fontsize=title_fontsize, pad=5)
        ax.set_xlabel(f"$\log_{{10}}$({xlabel})" if use_log else xlabel, fontsize=12)
        if use_log:
            ax.set_ylabel(r'Density per $\log_{10}$(unit)', fontsize=12)
        else:
            ax.set_ylabel('Density')

    for ax in axes[k:]:
        ax.axis('off')
    
    # ---------- Horizontal separators between rows ----------
    for row in range(1, rows):
        # Get all axes in the row above
        axes_above = axes[(row - 1)*cols_n : row*cols_n]

        # Find the bottom y-coordinate of the row above
        # We'll take the minimum y0 among axes in that row
        y_bottom = min(ax.get_position().y0 for ax in axes_above)

        # Create a line spanning the full figure width
        line = Line2D(
            [0, 1],             # x from left to right of figure
            [y_bottom - bar_padding - 0.0255],  # slightly below bottom of row
            transform=fig.transFigure,
            color='black',
            linewidth=1.5,
            clip_on=False
        )
        fig.add_artist(line)

    return fig, axes


### ------------ Connectivity & User Count histograms ------------ ###

# --- County-level SCI histograms ---
df_dedup = (
    df_sci
    .assign(
        loc_min=np.minimum(df_sci['user_loc'], df_sci['fr_loc']),
        loc_max=np.maximum(df_sci['user_loc'], df_sci['fr_loc'])
    )
    .drop_duplicates(subset=['loc_min', 'loc_max'])
    .drop(columns=['loc_min', 'loc_max'])
)
df_endo = (df_dedup[df_dedup['user_loc'] == df_dedup['fr_loc']]).copy()
df_exo = (df_dedup[df_dedup['user_loc'] != df_dedup['fr_loc']]).copy()

fig, axes = plot_histograms_with_pdfs_kde(
    dfs = [df_endo, df_exo, df_dedup],
    cols=['scaled_sci', 'scaled_sci', 'scaled_sci'],
    titles=
        ['$\log_{10}$(Inner-County SCI) per County', '$\log_{10}$(Outgoing County SCI) per County', 
         '$\log_{10}$(Total County SCI) per County'],
    x_labels=['Inner-County SCI', 'Outgoing County SCI', 'Total County SCI'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9,
    bin_max=65,
    kde_max=200000
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\connectivity\\sci\\sci_histograms.png", dpi=800, bbox_inches='tight')

# --- County-level Connection histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_inner_county]*6,
    cols=['inter_county_connections', 'outer_county_connections', 'total connections', 'user_est', 'pop_est', 'coverage est'],
    titles=
        ['$\log_{10}$(Inner-County Connections) per County', '$\log_{10}$(Outgoing County Connections) per County', 
         '$\log_{10}$(Total County Connections) per County', '$\log_{10}$(User Estimate) per County', 
         '$\log_{10}$(Population Estimate) per County', 'Coverage Estimate per County'],
    x_labels=['Inner-County Connections', 'Outgoing County Connections', 'Total County Connections', 
              'County User Estimates', 'County Population Estimates', 'County Coverage Estimates'],
    no_log_cols=['coverage est'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\connectivity\\connections\\county_connections_histograms.png", dpi=800, bbox_inches='tight')

# --- CBSA-level Connection, User, Population, & Coverage Estimate histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_cbsa]*6,
    cols=['total inter_cbsa connections', 'outer_cbsa_connections', 'total connections',
          'user_est', 'pop_est', 'coverage est'],
    titles=
        ['$\log_{10}$(Inner-CBSA Connections) per CBSA', '$\log_{10}$(Outgoing CBSA Connections) per CBSA', 
         '$\log_{10}$(Total CBSA Connections) per CBSA', '$\log_{10}$(User Estimate) per CBSA', 
         '$\log_{10}$(Population Estimate) per CBSA', 'Coverage Estimate per CBSA'],
    x_labels=['Inner-CBSA Connections', 'Outgoing CBSA Connections', 'Total CBSA Connections', 
              'CBSA User Estimates', 'CBSA Population Estimates', 'CBSA Coverage Estimates'],
    no_log_cols=['coverage est'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\connectivity\\connections\\cbsa_connections_histograms.png", dpi=800, bbox_inches='tight')

# --- MSA-level Connection, User, Population, & Coverage Estimate histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_cbsa]*6,
    cols=['total inter_cbsa connections', 'outer_cbsa_connections', 'total connections',
          'user_est', 'pop_est', 'coverage est'],
    titles=
        ['$\log_{10}$(Inner-MSA Connections) per MSA', '$\log_{10}$(Outgoing MSA Connections) per MSA', 
         '$\log_{10}$(Total MSA Connections) per MSA', '$\log_{10}$(User Estimate) per MSA', 
         '$\log_{10}$(Population Estimate) per MSA', 'Coverage Estimate per MSA'],
    x_labels=['Inner-MSA Connections', 'Outgoing MSA Connections', 'Total MSA Connections', 
              'MSA User Estimates', 'MSA Population Estimates', 'MSA Coverage Estimates'],
    no_log_cols=['coverage est'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\connectivity\\connections\\msa_connections_histograms.png", dpi=800, bbox_inches='tight')

# --- muSA-level Connection, User, Population, & Coverage Estimate histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_cbsa]*6,
    cols=['total inter_cbsa connections', 'outer_cbsa_connections', 'total connections',
          'user_est', 'pop_est', 'coverage est'],
    titles=
        ['$\log_{10}$(Inner-muSA Connections) per muSA', '$\log_{10}$(Outgoing muSA Connections) per muSA', 
         '$\log_{10}$(Total muSA Connections) per muSA', '$\log_{10}$(User Estimate) per muSA', 
         '$\log_{10}$(Population Estimate) per muSA', 'Coverage Estimate per muSA'],
    x_labels=['Inner-muSA Connections', 'Outgoing muSA Connections', 'Total muSA Connections', 
              'muSA User Estimates', 'muSA Population Estimates', 'muSA Coverage Estimates'],
    no_log_cols=['coverage est'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\connectivity\\connections\\musa_connections_histograms.png", dpi=800, bbox_inches='tight')

### ------------ Connectivity per Capita/User Count histograms ------------ ###

df_inner_county['inter_county_connections per capita'] = df_inner_county['inter_county_connections']/df_inner_county['pop_est']
df_inner_county['inter_county_connections per user'] = df_inner_county['inter_county_connections']/df_inner_county['user_est']

df_inner_county['outer_county_connections per capita'] = df_inner_county['outer_county_connections']/df_inner_county['pop_est']
df_inner_county['outer_county_connections per user'] = df_inner_county['outer_county_connections']/df_inner_county['user_est']

df_inner_county['total connections per capita'] = df_inner_county['total connections']/df_inner_county['pop_est']
df_inner_county['total connections per user'] = df_inner_county['total connections']/df_inner_county['user_est']


df_cbsa['inter_cbsa_connections per capita'] = df_cbsa['total inter_cbsa connections']/df_cbsa['pop_est']
df_cbsa['inter_cbsa_connections per user'] = df_cbsa['total inter_cbsa connections']/df_cbsa['user_est']

df_cbsa['outer_cbsa_connections per capita'] = df_cbsa['outer_cbsa_connections']/df_cbsa['pop_est']
df_cbsa['outer_cbsa_connections per user'] = df_cbsa['outer_cbsa_connections']/df_cbsa['user_est']

df_cbsa['total connections per capita'] = df_cbsa['total connections']/df_cbsa['pop_est']
df_cbsa['total connections per user'] = df_cbsa['total connections']/df_cbsa['user_est']


df_msa['inter_cbsa_connections per capita'] = df_msa['total inter_cbsa connections']/df_msa['pop_est']
df_msa['inter_cbsa_connections per user'] = df_msa['total inter_cbsa connections']/df_msa['user_est']

df_msa['outer_cbsa_connections per capita'] = df_msa['outer_cbsa_connections']/df_msa['pop_est']
df_msa['outer_cbsa_connections per user'] = df_msa['outer_cbsa_connections']/df_msa['user_est']

df_msa['total connections per capita'] = df_msa['total connections']/df_msa['pop_est']
df_msa['total connections per user'] = df_msa['total connections']/df_msa['user_est']


df_musa['inter_cbsa_connections per capita'] = df_musa['total inter_cbsa connections']/df_musa['pop_est']
df_musa['inter_cbsa_connections per user'] = df_musa['total inter_cbsa connections']/df_musa['user_est']

df_musa['outer_cbsa_connections per capita'] = df_musa['outer_cbsa_connections']/df_musa['pop_est']
df_musa['outer_cbsa_connections per user'] = df_musa['outer_cbsa_connections']/df_musa['user_est']

df_musa['total connections per capita'] = df_musa['total connections']/df_musa['pop_est']
df_musa['total connections per user'] = df_musa['total connections']/df_musa['user_est']

# --- County-level Connection histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_inner_county]*6,
    cols=['inter_county_connections per capita', 'outer_county_connections per capita', 'total connections per capita', 'inter_county_connections per user', 'outer_county_connections per user', 'total connections per user'],
    titles=
        ['$\log_{10}$(Inner-County Connections per Capita) per County', '$\log_{10}$(Outgoing County Connections per Capita) per County', 
         '$\log_{10}$(Total County Connections per Capita) per County', '$\log_{10}$(Inner-County Connections per User) per County', 
         '$\log_{10}$(Outgoing County Connections per User) per County', '$\log_{10}$(Total County Connections per User) per County',],
    x_labels=['Inner-County Connections per Capita', 'Outgoing County Connections per Capita', 'Total County Connections per Capita', 
              'Inner-County Connections per User', 'Outgoing County Connections per User', 'Total County Connections per User'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\per_capita & per_user\\county_connections_histograms.png", dpi=800, bbox_inches='tight')

# --- CBSA-level Connection, User, Population, & Coverage Estimate histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_cbsa]*6,
    cols=['inter_cbsa_connections per capita', 'outer_cbsa_connections per capita', 'total connections per capita',
          'inter_cbsa_connections per user', 'outer_cbsa_connections per user', 'total connections per user'],
    titles=
        ['$\log_{10}$(Inner-CBSA Connections per Capita) per CBSA', '$\log_{10}$(Outgoing CBSA Connections per Capita) per CBSA', 
         '$\log_{10}$(Total CBSA Connections per Capita) per CBSA', '$\log_{10}$(Inner-CBSA Connections per User) per CBSA', 
         '$\log_{10}$(Outgoing CBSA Connections per User) per CBSA', '$\log_{10}$(Total CBSA Connections per User) per CBSA',],
    x_labels=['Inner-CBSA Connections per Capita', 'Outgoing CBSA Connections per Capita', 'Total CBSA Connections per Capita', 
              'Inner-CBSA Connections per User', 'Outgoing CBSA Connections per User', 'Total CBSA Connections per User'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\per_capita & per_user\\cbsa_connections_histograms.png", dpi=800, bbox_inches='tight')

# --- MSA-level Connection, User, Population, & Coverage Estimate histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_cbsa]*6,
    cols=['inter_cbsa_connections per capita', 'outer_cbsa_connections per capita', 'total connections per capita',
          'inter_cbsa_connections per user', 'outer_cbsa_connections per user', 'total connections per user'],
    titles=
        ['$\log_{10}$(Inner-MSA Connections per Capita) per MSA', '$\log_{10}$(Outgoing MSA Connections per Capita) per MSA', 
         '$\log_{10}$(Total MSA Connections per Capita) per MSA', '$\log_{10}$(Inner-MSA Connections per User) per MSA', 
         '$\log_{10}$(Outgoing MSA Connections per User) per MSA', '$\log_{10}$(Total MSA Connections per User) per MSA',],
     x_labels=['Inner-MSA Connections per Capita', 'Outgoing MSA Connections per Capita', 'Total MSA Connections per Capita', 
              'Inner-MSA Connections per User', 'Outgoing MSA Connections per User', 'Total MSA Connections per User'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\per_capita & per_user\\msa_connections_histograms.png", dpi=800, bbox_inches='tight')

# --- muSA-level Connection, User, Population, & Coverage Estimate histograms ---
fig, axes = plot_histograms_with_pdfs_kde(
    dfs=[df_cbsa]*6,
    cols=['inter_cbsa_connections per capita', 'outer_cbsa_connections per capita', 'total connections per capita',
          'inter_cbsa_connections per user', 'outer_cbsa_connections per user', 'total connections per user'],
    titles=
        ['$\log_{10}$(Inner-muSA Connections per Capita) per muSA', '$\log_{10}$(Outgoing muSA Connections per Capita) per muSA', 
         '$\log_{10}$(Total muSA Connections per Capita) per muSA', '$\log_{10}$(Inner-muSA Connections per User) per muSA', 
         '$\log_{10}$(Outgoing muSA Connections per User) per muSA', '$\log_{10}$(Total muSA Connections per User) per muSA',],
    x_labels=['Inner-muSA Connections per Capita', 'Outgoing muSA Connections per Capita', 'Total muSA Connections per Capita', 
              'Inner-muSA Connections per User', 'Outgoing muSA Connections per User', 'Total muSA Connections per User'],
    log_x=True,
    figsize_per_plot=5,
    bar_padding=0.02,
    title_fontsize=16,
    stats_fontsize=8,
    legend_fontsize=9
)
plt.show()
fig.savefig("F:\\dsl_CLIMA\\submittable\\plots\\histograms\\per_capita & per_user\\musa_connections_histograms.png", dpi=800, bbox_inches='tight')