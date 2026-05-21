"""
py_chart.py
===========
Side-by-side educational-attainment and household-income pie charts for
the Hamilton Beach and Red Hook focus communities.

Uses the IPUMS NHGIS ``nhgis0004_ds267_20235`` ACS 5-year extract at
block-group resolution. Educational attainment is collapsed from the 24
ACS bins (``ASP3E002``..``ASP3E025``) into 9 readable categories.
Household income is collapsed from 16 ACS bins
(``ASQOE002``..``ASQOE017``) into 9 income brackets.

Outputs (written to ``plots/demographics/pi charts/``):

- ``hamBeach_pi.png``: Hamilton Beach education + income, green palette.
- ``redHook_pi.png``: Red Hook education + income, blue palette.

The two output figures use a consistent legend / category ordering so
the distributions are directly comparable side by side.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

# Load the ACS block-group extract.
df = pd.read_csv(
    r'F:\dsl_CLIMA\projects\submittable\clima\source\census demographic files\nhgis0004_ds267_20235_blck_grp.csv',
    dtype={'TL_GEO_ID': str}
)

# Target block-group GEOIDs.
# Hamilton Beach is a single block group (Queens, NY).
# Red Hook spans 11 block groups in Brooklyn, NY.
ham_beach = ['360810884006']
red_hook = [
    '360470053031', '360470053011', '360470053012', '360470053022',
    '360470085003', '360470059002', '360470059001', '360470085001',
    '360470085002', '360470053021', '360470053023'
]

ham_df = df[df['TL_GEO_ID'].isin(ham_beach)]
red_df = df[df['TL_GEO_ID'].isin(red_hook)]

# ----------------------------------------------------------------------
# Raw ACS bin column lists.
# - ASP3E002..ASP3E025: 24 educational-attainment bins for age 25+.
# - ASQOE002..ASQOE017: 16 household-income brackets.
# ----------------------------------------------------------------------
edu_cols = [f"ASP3E{str(i).zfill(3)}" for i in range(2, 26)]
income_cols = [f"ASQOE{str(i).zfill(3)}" for i in range(2, 18)]

ham_edu_dist = ham_df[edu_cols].sum()
red_edu_dist = red_df[edu_cols].sum()
ham_income_dist = ham_df[income_cols].sum()
red_income_dist = red_df[income_cols].sum()

# ----------------------------------------------------------------------
# Education bucket mapping: collapse the 24 ACS bins into 9 readable
# categories. Indices reference positions within ``edu_cols``.
# ----------------------------------------------------------------------
simplified_edu_groups = {
    "No HS Diploma": list(range(0, 15)),
    "Only HS Diploma": [15],
    "GED": [16],
    "Incomplete College Degree": [17, 18],
    "Associate": [19],
    "Bachelor": [20],
    "Master": [21],
    "Professional": [22],
    "Doctorate": [23]
}


def simplify_education_distribution(data):
    """Collapse the 24 ACS education bins into 9 readable categories.

    Parameters
    ----------
    data : pandas.Series
        Series with one entry per raw ASP3E bin (length 24).

    Returns
    -------
    pandas.Series
        Series with one entry per simplified bucket, indexed by the
        bucket label.
    """
    simplified = {}
    for label, indices in simplified_edu_groups.items():
        simplified[label] = data.iloc[indices].sum()
    return pd.Series(simplified)


ham_edu_simple = simplify_education_distribution(ham_edu_dist)
red_edu_simple = simplify_education_distribution(red_edu_dist)

# ----------------------------------------------------------------------
# Income bucket mapping: collapse the 16 ACS bins into 9 brackets.
# Indices reference positions within ``income_cols``.
# ----------------------------------------------------------------------
merged_income_labels = [
    "<$10k", "$10k-25k", "$25k-35k", "$35k-45k", "$45k-60k",
    "$60k-85k", "$100k-150k", "$150k-200k", "$200k+"
]

merged_income_groups = {
    "<$10k": [0],                    # ASQOE002
    "$10k-25k": [1, 2, 3],           # ASQOE003-ASQOE005
    "$25k-35k": [4, 5],              # ASQOE006-ASQOE007
    "$35k-45k": [6, 7],              # ASQOE008-ASQOE009
    "$45k-60k": [8, 9],              # ASQOE010-ASQOE011
    "$60k-85k": [10, 11],            # ASQOE012-ASQOE013
    "$100k-150k": [12, 13],          # ASQOE014-ASQOE015
    "$150k-200k": [14],              # ASQOE016
    "$200k+": [15]                   # ASQOE017
}


def simplify_income_distribution(data):
    """Collapse the 16 ACS income bins into 9 brackets.

    Parameters
    ----------
    data : pandas.Series
        Series with one entry per raw ASQOE bin (length 16).

    Returns
    -------
    pandas.Series
        Series with one entry per income bracket, indexed by bracket
        label.
    """
    simplified = {}
    for label, indices in merged_income_groups.items():
        simplified[label] = data.iloc[indices].sum()
    return pd.Series(simplified)


ham_income_simple = simplify_income_distribution(ham_income_dist)
red_income_simple = simplify_income_distribution(red_income_dist)


def plot_pie(data, labels, title, palette='Blues', ax=None, radius=1.6):
    """Render a single annotated pie chart with percentage labels.

    Slices are ordered along a fixed logical ordering (smallest-to-
    largest education tier or smallest-to-largest income bracket) so
    color and label position are consistent across plots that share a
    palette. Zero-count categories are omitted from the pie but kept in
    the legend so the visual key matches across communities.

    Parameters
    ----------
    data : pandas.Series
        Counts indexed by the bucket labels in ``labels``.
    labels : list of str
        Full list of bucket labels expected in ``data``.
    title : str
        Subplot title.
    palette : str, default 'Blues'
        Matplotlib colormap name. Hamilton Beach uses 'Greens', Red
        Hook uses 'Blues'.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure is created.
    radius : float, default 1.6
        Pie radius in data units.
    """
    data = data.fillna(0)

    if len(data) != len(labels):
        print(f"Warning: Mismatch between data and labels for {title}")
        return

    # Logical orderings for the two figure types. Hispanic and "all"
    # buckets are filtered out automatically because they don't appear
    # in either list.
    edu_order = [
        "No HS Diploma", "Only HS Diploma", "GED", "Incomplete College Degree",
        "Associate", "Bachelor", "Master", "Professional", "Doctorate"
    ]
    income_order = [
        "<$10k", "$10k-25k", "$25k-35k", "$35k-45k", "$45k-60k",
        "$60k-85k", "$100k-150k", "$150k-200k", "$200k+"
    ]
    if all(label in edu_order for label in labels):
        logical_order = edu_order
    elif all(label in income_order for label in labels):
        logical_order = income_order
    else:
        logical_order = sorted(labels)

    # Build a stable label-to-color mapping so the same bucket always
    # gets the same color across panels.
    cmap = cm.get_cmap(palette, len(logical_order))
    label_color_map = {label: cmap(i) for i, label in enumerate(logical_order)}

    # Slice the pie only over non-zero buckets so we don't emit
    # invisible wedges that still consume legend space.
    data_nonzero = data[data > 0]
    sorted_labels = [label for label in logical_order if label in data_nonzero.index]
    sorted_data = [data[label] for label in sorted_labels]
    sorted_colors = [label_color_map[label] for label in sorted_labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    if not sorted_data:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
        ax.axis('off')
        return

    wedges, _ = ax.pie(
        sorted_data,
        colors=sorted_colors,
        startangle=0,
        labels=None,
        radius=radius,
        wedgeprops=dict(edgecolor='white')
    )

    total = sum(sorted_data)

    # Annotate each wedge with its percentage, anchored by a thin
    # leader line so labels don't collide with the pie edge.
    for wedge, val in zip(wedges, sorted_data):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        ha = 'left' if x > 0 else 'right'
        pct = f"{100 * val / total:.1f}%"

        label_pos = radius * 1.15
        arrow_start = radius * 0.75

        ax.annotate(
            pct,
            xy=(x * arrow_start, y * arrow_start),
            xytext=(x * label_pos, y * label_pos),
            ha=ha,
            va='center',
            fontsize=18,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
            color='black'
        )

    # Build the legend from the full bucket list (not just non-zero) so
    # zero-count categories still appear in the visual key.
    legend_labels = [label for label in logical_order if label in labels]
    legend_patches = [
        Patch(facecolor=label_color_map[label], edgecolor='none') for label in legend_labels
    ]

    ax.legend(
        legend_patches,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        fontsize=13,
        labelcolor='black'
    )

    ax.set_title(title, fontsize=24, pad=30)
    ax.axis('equal')


edu_labels = [
    "No HS", "Only HS", "GED", "Incomplete College Degree",
    "Associate", "Bachelor", "Master", "Professional", "Doctorate"
]
income_labels = [
    "<$10k", "$10k-25k", "$25k-35k", "$35k-45k", "$45k-60k",
    "$60k-85k", "$100k-150k", "$150k-200k", "$200k+"
]

# ----------------------------------------------------------------------
# Hamilton Beach: education + income side by side, green palette.
# ----------------------------------------------------------------------
fig_ham, axes_ham = plt.subplots(1, 2, figsize=(14, 7))
plot_pie(ham_edu_simple, ham_edu_simple.index.tolist(), "Educational Level", palette='Greens', ax=axes_ham[0])
plot_pie(ham_income_simple, ham_income_simple.index.tolist(), "Household Income", palette='Greens', ax=axes_ham[1])
fig_ham.suptitle('Hamilton Beach Demographics', fontsize=40, x=.525)

plt.tight_layout()
plt.show()
fig_ham.savefig('F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\pi charts\\hamBeach_pi.png', dpi=800, bbox_inches='tight')

# ----------------------------------------------------------------------
# Red Hook: education + income side by side, blue palette. The category
# axes are kept identical to the Hamilton Beach figure so the two are
# directly comparable.
# ----------------------------------------------------------------------
fig_red, axes_red = plt.subplots(1, 2, figsize=(14, 7))
plot_pie(red_edu_simple, red_edu_simple.index.tolist(), "Educational Level", palette='Blues', ax=axes_red[0])
plot_pie(red_income_simple, red_income_simple.index.tolist(), "Household Income", palette='Blues', ax=axes_red[1])
fig_red.suptitle('Red Hook Demographics', fontsize=40, x=.525)

plt.tight_layout()
plt.show()
fig_red.savefig('F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\pi charts\\redHook_pi.png', dpi=800, bbox_inches='tight')
