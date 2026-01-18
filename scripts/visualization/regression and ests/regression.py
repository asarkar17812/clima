import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats
import geopandas as gpd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

df_inner_county = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_outer_county.csv')
df_cbsa = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_cbsa.csv')
df_msa = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_msa.csv')
df_musa = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_musa.csv')

def log_log_regression_plot_on_ax(df, x_col, y_col, ax, title=None, region=''):
    df_plot = df[[x_col, y_col, 'user_est', 'pop_est']].copy()
    df_plot = df_plot[(df_plot[x_col] > 0) & (df_plot[y_col] > 0)]
    N = len(df_plot)

    if N < 2:
        raise ValueError("Not enough positive entries to fit regression.")

    x = np.log10(df_plot[x_col].values)
    y = np.log10(df_plot[y_col].values)

    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()

    slope = results.params[1]
    intercept = results.params[0]
    r2 = results.rsquared

    se = results.bse
    t_val = stats.t.ppf(0.975, df=N - 2)

    slope_ci = slope + np.array([-1, 1]) * t_val * se[1]
    intercept_ci = intercept + np.array([-1, 1]) * t_val * se[0]

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = intercept + slope * x_fit

    y_pred_fit = results.get_prediction(sm.add_constant(x_fit))
    y_ci = y_pred_fit.conf_int(alpha=0.05)

    y_hat = results.predict(x_with_const)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    rse = np.sqrt(np.sum((y - y_hat) ** 2) / (N - 2))

    # Statistics
    mean_x, mean_y = np.mean(x), np.mean(y)
    median_x, median_y = np.median(x), np.median(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    # Mode approximation for x
    hist, bin_edges = np.histogram(x, bins='auto')
    mode_bin = np.argmax(hist)
    mode_x = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
    mode_y = intercept + slope * mode_x

    # Plot points
    ax.scatter(10**x, 10**y, color='darkblue', alpha=0.6, label='Data points')
    ax.scatter(10**mean_x, 10**mean_y, color='fuchsia', s=80, label='Mean')
    ax.scatter(10**median_x, 10**median_y, color='orange', s=80, label='Median')
    ax.scatter(10**mode_x, 10**mode_y, color='lime', s=80, label='Mode (approx.)')

    # Regression line and confidence interval
    ax.plot(10**x_fit, 10**y_fit, color='red', lw=2, label='Regression line')
    ax.fill_between(
        10**x_fit,
        10**y_ci[:, 0],
        10**y_ci[:, 1],
        color='darkred',
        alpha=0.2,
        label='95% Conf. Int.'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(
        rf'$\log_{{10}}\!\left(\frac{{N_i}}{{\langle N_{{\mathrm{{{region}}}}}\rangle}}\right)$ (Normed Pop. by {region})',
        fontsize=18
    )
    ax.set_ylabel(
        rf'$\log_{{10}}\!\left(\frac{{K_{{r,\mathrm{{total}}}}}}{{\langle K_{{r,\mathrm{{{region}}}}}\rangle}}\right)$ (Normed & Rescaled by {region})',
        fontsize=18
    )

    if title:
        ax.set_title(title, fontsize=24)

    ax.tick_params(axis='both', labelsize=10)

    # Legend/statistics text
    legend_text = (
        f"β (slope): {slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]\n"
        f"γ (intercept): {intercept:.3f} [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]\n"
        f"R²: {r2:.3f}\n"
        f"RMSE: {rmse:.3f}\n"
        f"RSE: {rse:.3f}\n"
        f"Average User Est.: {df_plot['user_est'].mean():,.0f}\n"
        f"Average Pop Est.: {df_plot['pop_est'].mean():,.0f}\n"
        f"# of GEOIDs: {N}"
    )

    ax.text(
        0.05, 0.95, legend_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8)
    )

    ax.legend(fontsize=12, loc='lower right')


fig, axs = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)
    
log_log_regression_plot_on_ax(
    df=df_inner_county,
    x_col='normed pop_est',
    y_col='rescaled inter_county_connections',
    ax=axs[0],
    title='Inter-County Connections by County',
    region='County'
)

log_log_regression_plot_on_ax(
    df=df_inner_county,
    x_col='normed pop_est',
    y_col='rescaled outer_county_connections',
    ax=axs[1],
    title='Outgoing Connections by County',
    region='County'
)

log_log_regression_plot_on_ax(
    df=df_inner_county,
    x_col='normed pop_est',
    y_col='rescaled total connections',
    ax=axs[2],
    title='Total Connections by County',
    region='County'
)

plt.tight_layout()
plt.show()
# fig.savefig("F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\regressions\\county_connection_regressions.png", dpi=800, bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)

log_log_regression_plot_on_ax(
    df=df_cbsa,
    x_col='normed pop_est',
    y_col='rescaled total inter_cbsa connections',
    ax=axs[0],
    title='Inter-County Connections by CBSA',
    region='CBSA'
)

log_log_regression_plot_on_ax(
    df=df_cbsa,
    x_col='normed pop_est',
    y_col='rescaled outer_cbsa_connections',
    ax=axs[1],
    title='Outgoing Connections by CBSA',
    region='CBSA'
)

log_log_regression_plot_on_ax(
    df=df_cbsa,
    x_col='normed pop_est',
    y_col='rescaled total connections',
    ax=axs[2],
    title='Total Connections by CBSA',
    region='CBSA'
)

plt.tight_layout()
plt.show()
# fig.savefig("F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\regressions\\cbsa_connection_regressions.png", dpi=800, bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)

log_log_regression_plot_on_ax(
    df=df_msa,
    x_col='normed pop_est',
    y_col='rescaled total inter_cbsa connections',
    ax=axs[0],
    title='Inter-County Connections by MSA',
    region='MSA'
)

log_log_regression_plot_on_ax(
    df=df_msa,
    x_col='normed pop_est',
    y_col='rescaled outer_cbsa_connections',
    ax=axs[1],
    title='Outgoing Connections by MSA',
    region='MSA'
)

log_log_regression_plot_on_ax(
    df=df_msa,
    x_col='normed pop_est',
    y_col='rescaled total connections',
    ax=axs[2],
    title='Total Connections by MSA',
    region='MSA'
)

plt.tight_layout()
plt.show()
# fig.savefig("F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\regressions\\msa_connection_regressions.png", dpi=800, bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)

log_log_regression_plot_on_ax(
    df=df_musa,
    x_col='normed pop_est',
    y_col='rescaled total inter_cbsa connections',
    ax=axs[0],
    title='Inter-County Connections by muSA',
    region='muSA'
)

log_log_regression_plot_on_ax(
    df=df_musa,
    x_col='normed pop_est',
    y_col='rescaled outer_cbsa_connections',
    ax=axs[1],
    title='Outgoing Connections by muSA',
    region='muSA'
)

log_log_regression_plot_on_ax(
    df=df_musa,
    x_col='normed pop_est',
    y_col='rescaled total connections',
    ax=axs[2],
    title='Total Connections by muSA',
    region='muSA'
)

plt.tight_layout()
plt.show()
# fig.savefig("F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\regressions\\musa_connection_regressions.png", dpi=800, bbox_inches='tight')