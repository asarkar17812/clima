
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

# Load dataset
df = pd.read_csv(r'F:\dsl_CLIMA\projects\submittable\clima\source\census demographic files\nhgis0004_ds267_20235_blck_grp.csv', dtype={'TL_GEO_ID': str})

# GEOID groups
ham_beach = ['360810884006']
red_hook = [
    '360470053031', '360470053011', '360470053012', '360470053022',
    '360470085003', '360470059002', '360470059001', '360470085001',
    '360470085002', '360470053021', '360470053023'
]

# Filter data
ham_df = df[df['TL_GEO_ID'].isin(ham_beach)]
red_df = df[df['TL_GEO_ID'].isin(red_hook)]

# Education and income columns
edu_cols = [f"ASP3E{str(i).zfill(3)}" for i in range(2, 26)]
income_cols = [f"ASQOE{str(i).zfill(3)}" for i in range(2, 18)]

# Sum values
ham_edu_dist = ham_df[edu_cols].sum()
red_edu_dist = red_df[edu_cols].sum()
ham_income_dist = ham_df[income_cols].sum()
red_income_dist = red_df[income_cols].sum()

# Simplified education mapping
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
    simplified = {}
    for label, indices in simplified_edu_groups.items():
        simplified[label] = data.iloc[indices].sum()
    return pd.Series(simplified)

# Apply simplification
ham_edu_simple = simplify_education_distribution(ham_edu_dist)
red_edu_simple = simplify_education_distribution(red_edu_dist)

merged_income_labels = [
    "<$10k", "$10k–25k", "$25k–35k", "$35k–45k", "$45k–60k",
    "$60k–85k", "$100k–150k", "$150k–200k", "$200k+"
]

# Mapping from ASQOE002–ASQOE017 (i.e., index 0–15)
merged_income_groups = {
    "<$10k": [0],                    # ASQOE002
    "$10k–25k": [1, 2, 3],           # ASQOE003 to ASQOE005
    "$25k–35k": [4, 5],              # ASQOE006 to ASQOE007
    "$35k–45k": [6, 7],              # ASQOE008 to ASQOE009
    "$45k–60k": [8, 9],              # ASQOE010 to ASQOE011
    "$60k–85k": [10, 11],            # ASQOE012 to ASQOE013
    "$100k–150k": [12, 13],          # ASQOE014 to ASQOE015
    "$150k–200k": [14],              # ASQOE016
    "$200k+": [15]                   # ASQOE017
}

def simplify_income_distribution(data):
    simplified = {}
    for label, indices in merged_income_groups.items():
        simplified[label] = data.iloc[indices].sum()
    return pd.Series(simplified)

ham_income_simple = simplify_income_distribution(ham_income_dist)
red_income_simple = simplify_income_distribution(red_income_dist)

def plot_pie(data, labels, title, palette='Blues', ax=None, radius=1.6):
    data = data.fillna(0)

    if len(data) != len(labels):
        print(f"Warning: Mismatch between data and labels for {title}")
        return

    # Logical orderings
    edu_order = [
        "No HS Diploma", "Only HS Diploma", "GED", "Incomplete College Degree",
        "Associate", "Bachelor", "Master", "Professional", "Doctorate"
    ]
    income_order = [
        "<$10k", "$10k–25k", "$25k–35k", "$35k–45k", "$45k–60k",
        "$60k–85k", "$100k–150k", "$150k–200k", "$200k+"
    ]
    if all(label in edu_order for label in labels):
        logical_order = edu_order
    elif all(label in income_order for label in labels):
        logical_order = income_order
    else:
        logical_order = sorted(labels)

    # Colormap and full label-color mapping
    cmap = cm.get_cmap(palette, len(logical_order))
    label_color_map = {label: cmap(i) for i, label in enumerate(logical_order)}

    # Data and colors for non-zero slices
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

    # Legend: all categories, not just shown ones
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
    # plt.subplots_adjust(left=0.05, right=0.65, top=0.85, bottom=0.05)


edu_labels = [
        "No HS", "Only HS", "GED", "Incomplete College Degree",
        "Associate", "Bachelor", "Master", "Professional", "Doctorate"
    ]
income_labels = [
        "<$10k", "$10k–25k", "$25k–35k", "$35k–45k", "$45k–60k",
        "$60k–85k", "$100k–150k", "$150k–200k", "$200k+"
    ]
# Hamilton Beach plots side-by-side (with smaller radius)
fig_ham, axes_ham = plt.subplots(1, 2, figsize=(14, 7))
plot_pie(ham_edu_simple, ham_edu_simple.index.tolist(), "Educational Level", palette='Greens', ax=axes_ham[0])
plot_pie(ham_income_simple, ham_income_simple.index.tolist(), "Household Income", palette='Greens', ax=axes_ham[1])
fig_ham.suptitle('Hamilton Beach Demographics', fontsize=40, x=.525)

plt.tight_layout()
plt.show()
fig_ham.savefig('F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\pi charts\\hamBeach_pi.png', dpi=800, bbox_inches='tight')
# Red Hook plots side-by-side (default radius)
fig_red, axes_red = plt.subplots(1, 2, figsize=(14, 7))
plot_pie(red_edu_simple, red_edu_simple.index.tolist(), "Educational Level", palette='Blues', ax=axes_red[0])
plot_pie(red_income_simple, red_income_simple.index.tolist(), "Household Income", palette='Blues', ax=axes_red[1])
fig_red.suptitle('Red Hook Demographics', fontsize=40, x=.525)

plt.tight_layout()
plt.show()
fig_red.savefig('F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\pi charts\\redHook_pi.png', dpi=800, bbox_inches='tight')