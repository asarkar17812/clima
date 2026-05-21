"""
demographics.py
===============
Census-block demographic figures for the Hamilton Beach + Howard Beach
focus communities (FIPS prefix ``360810884006``), with NYC-wide
census-block averages overlaid for comparison.

Data sources (read from ``source/census demographic files/``):

- ``DECENNIALDHC2020.H3-Data.csv``: housing occupancy counts (H3 table).
- ``DECENNIALDHC2020.H4-Data.csv``: housing tenure counts (H4 table).
- ``DECENNIALDHC2020.H9-Data.csv``: household size counts (H9 table).
- ``DECENNIALDHC2020.P3-Data.csv``: race counts (P3 table).
- ``DECENNIALDHC2020.P5-Data.csv``: Hispanic or Latino origin by race
  (P5 table).
- ``DECENNIALDHC2020.P12-Data.csv``: sex by age across the total
  population (P12 table).

For each target block, and for the aggregate over all target blocks,
the script produces:

- Age / sex population pyramids.
- Race / ethnicity distributions (White non-Hispanic, Black non-Hispanic,
  Asian non-Hispanic, Hispanic of any race).
- Household size distributions (1, 2, 3, 4, 5+ persons).
- Housing occupancy (occupied vs. vacant).
- Housing tenure (owner-occupied vs. renter-occupied).

Outputs are written under ``plots/demographics/bar charts/`` in
sub-folders by figure type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Load every relevant 2020 Decennial Census DHC file. GEO_ID and NAME
# are read as strings so the 22-character block identifiers and full
# block names survive intact.
# ----------------------------------------------------------------------
df_tenure = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\census demographic files\\DECENNIALDHC2020.H4-Data.csv', dtype={'GEO_ID': str, 'NAME': str})
df_household_size = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\census demographic files\\DECENNIALDHC2020.H9-Data.csv', dtype={'GEO_ID': str, 'NAME': str})
df_sex_by_age = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\census demographic files\\DECENNIALDHC2020.P12-Data.csv', dtype={'GEO_ID': str, 'NAME': str})
df_race = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\census demographic files\\DECENNIALDHC2020.P3-Data.csv', dtype={'GEO_ID': str, 'NAME': str})
df_hispanic = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\census demographic files\\DECENNIALDHC2020.P5-Data.csv', dtype={'GEO_ID': str, 'NAME': str})
df_occupancy = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\census demographic files\\DECENNIALDHC2020.H3-Data.csv', dtype={'GEO_ID': str, 'NAME': str})

# Drop the duplicate ``NAME`` columns from every file but the first so
# the outer merge below doesn't create _x / _y collisions on NAME.
df_household_size = df_household_size.drop(columns=['NAME'])
df_sex_by_age = df_sex_by_age.drop(columns=['NAME'])
df_race = df_race.drop(columns=['NAME'])
df_hispanic = df_hispanic.drop(columns=['NAME'])
df_occupancy = df_occupancy.drop(columns=['NAME'])

# Outer-merge every table on GEO_ID so each block has all six DHC
# tables in one row. astype(str) at each step preserves the GEOID dtype
# through the merges.
df_merged = df_tenure.merge(df_household_size, on='GEO_ID', how='outer').astype(str) \
               .merge(df_sex_by_age, on='GEO_ID', how='outer').astype(str) \
               .merge(df_race, on='GEO_ID', how='outer').astype(str) \
               .merge(df_hispanic, on='GEO_ID', how='outer').astype(str) \
               .merge(df_occupancy, on='GEO_ID', how='outer').astype(str)
df_merged = df_merged.drop(columns=['Unnamed: 6', 'Unnamed: 10_x', 'Unnamed: 51', 'Unnamed: 10_y', 'Unnamed: 19', 'Unnamed: 5'])

# Target census blocks in Hamilton Beach / Howard Beach (Queens).
# All share the FIPS prefix 360810884006 (state=36 NY, county=081 Queens,
# tract=088400, block group=6).
target_locations = [
    '1000000US360810884006000', '1000000US360810884006001', '1000000US360810884006003',
    '1000000US360810884006005', '1000000US360810884006004', '1000000US360810884006006',
    '1000000US360810884006008', '1000000US360810884006009', '1000000US360810884006010'
]

df_filtered = df_merged.copy()
df_filtered = df_filtered[df_filtered['GEO_ID'].isin(target_locations)]

df_filtered.columns = df_filtered.columns.str.strip()
df_merged.columns = df_merged.columns.str.strip()


def plot_population_pyramid(df_subset, df_all):
    """Plot an aggregate age / sex pyramid for the target census blocks.

    Bars show the target-block percentage of total target-block
    population in each age bucket, split into male (left, blue) and
    female (right, pink). NYC-wide census-block averages are overlaid
    as dots for direct comparison.

    Parameters
    ----------
    df_subset : pandas.DataFrame
        DHC rows for the target census blocks.
    df_all : pandas.DataFrame
        DHC rows for all NYC census blocks (used to compute the citywide
        averages).
    """
    # P12 male age columns: P12_003N..P12_025N (23 buckets).
    # P12 female age columns: P12_027N..P12_049N (23 buckets).
    male_cols = [f'P12_{i:03}N' for i in range(3, 26)]
    female_cols = [f'P12_{i:03}N' for i in range(27, 50)]

    age_labels = [
        'Under 5', '5-9', '10-14', '15-17', '18-19', '20', '21', '22-24',
        '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
        '60-61', '62-64', '65-66', '67-69', '70-74', '75-79', '80-84', '85+'
    ]
    y = np.arange(len(age_labels))

    # Coerce numeric. The merge step left everything as strings.
    cols_to_convert = male_cols + female_cols + ['P12_001N']
    df_all[cols_to_convert] = df_all[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_subset[cols_to_convert] = df_subset[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Target-block age percentages.
    male_totals = df_subset[male_cols].sum()
    female_totals = df_subset[female_cols].sum()
    total_subset = male_totals.sum() + female_totals.sum()
    male_percent = (male_totals / total_subset) * 100
    female_percent = (female_totals / total_subset) * 100

    # NYC-wide averages, restricting to non-empty blocks.
    df_nonzero = df_all[df_all['P12_001N'] > 0].copy()
    total_all = df_nonzero[male_cols + female_cols].sum().sum()
    male_avg = df_nonzero[male_cols].sum() / total_all * 100
    female_avg = df_nonzero[female_cols].sum() / total_all * 100

    max_male = max(male_percent.max(), male_avg.max()) * 1.1
    max_female = max(female_percent.max(), female_avg.max()) * 1.1

    fig, ax = plt.subplots(figsize=(14, 9))

    # First pass: render the age labels invisibly so we can measure
    # their width in data coordinates and reserve a center channel that
    # the bars won't overlap.
    txts = []
    for i, label in enumerate(age_labels):
        txt = ax.text(0, i, label, ha='center', va='center', fontsize=16, fontweight='bold',
                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
        txts.append(txt)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    label_widths = []
    for txt in txts:
        bbox = txt.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        left_data = inv.transform((bbox.x0, 0))[0]
        right_data = inv.transform((bbox.x1, 0))[0]
        width_data = right_data - left_data
        label_widths.append(width_data)

    max_label_width = max(label_widths)
    center_padding = max_label_width + 0.5

    ax.cla()

    # Male bars extend left from -center_padding; female bars extend
    # right from +center_padding. The gap in between holds the age
    # labels.
    ax.barh(y, -male_percent.values, color='steelblue', alpha=0.7, label='Percent Male',
            left=-center_padding)
    ax.barh(y, female_percent.values, color='lightcoral', alpha=0.7, label='Percent Female',
            left=center_padding)

    # NYC-average reference dots, shifted out by the center padding so
    # they align with their respective sides.
    ax.plot(-male_avg.values - center_padding, y, 'o', color='navy', label='NYC Census Block Avg Male')
    ax.plot(female_avg.values + center_padding, y, 'o', color='darkred', label='NYC Census Block Avg Female')

    # Age labels rendered in the center channel.
    for i, label in enumerate(age_labels):
        ax.text(0, i, label, ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    # Vertical baselines at the two true zero lines.
    ax.vlines(-center_padding, ymin=-1, ymax=len(age_labels), colors='black', linewidth=1.5)
    ax.vlines(center_padding, ymin=-1, ymax=len(age_labels), colors='black', linewidth=1.5)

    ax.set_xlim(-max_male - center_padding * 2, max_female + center_padding * 2)
    ax.set_ylim(-1, len(age_labels))

    # Mirrored, padding-adjusted x-axis ticks.
    left_ticks = np.linspace(0, max_male, 6)
    right_ticks = np.linspace(0, max_female, 6)
    ticks_left = -left_ticks - center_padding
    ticks_right = right_ticks + center_padding
    all_ticks = np.concatenate((ticks_left[::-1], [0], ticks_right))
    ax.set_xticks(all_ticks)

    tick_labels = [f"{int(abs(x + center_padding))}%" if x < 0 else "" for x in ticks_left[::-1]]
    tick_labels += [""]
    tick_labels += [f"{int(x - center_padding)}%" if x > 0 else "" for x in ticks_right]
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Percentage of Total Population (%)", fontsize=16)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    ax.set_yticks([])

    ax.axvline(0, color='black', linewidth=1)

    ax.text(-max_male - center_padding, len(age_labels) + 0.5, "Male", ha='center',
            fontsize=12, fontweight='bold', color='steelblue')
    ax.text(max_female + center_padding, len(age_labels) + 0.5, "Female", ha='center',
            fontsize=12, fontweight='bold', color='darkred')

    ax.set_title("Age/Sex Population Distribution for Hamilton Beach Census Blocks", fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    fig.savefig("F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\allBlocks\\hb_allBlocks_popPyramid.png", dpi=800, bbox_inches='tight')


plot_population_pyramid(df_filtered, df_merged)


def plot_population_pyramid_by_block(df_subset, df_all):
    """Same as ``plot_population_pyramid`` but emits one figure per block.

    Each block's percentages are normalized by that block's own total
    population, so the bars are directly comparable across blocks even
    when the absolute populations differ.

    Parameters
    ----------
    df_subset : pandas.DataFrame
        DHC rows for the target census blocks.
    df_all : pandas.DataFrame
        DHC rows for all NYC census blocks (citywide averages).
    """
    male_cols = [f'P12_{i:03}N' for i in range(3, 26)]
    female_cols = [f'P12_{i:03}N' for i in range(27, 50)]

    age_labels = [
        'Under 5', '5-9', '10-14', '15-17', '18-19', '20', '21', '22-24',
        '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
        '60-61', '62-64', '65-66', '67-69', '70-74', '75-79', '80-84', '85+'
    ]
    y = np.arange(len(age_labels))

    label_fontsize = 16
    # Fixed center channel width; simpler than the dynamic measurement
    # in the aggregate version because per-block plots are smaller.
    center_pad = 2

    cols_to_convert = male_cols + female_cols + ['P12_001N']
    df_all[cols_to_convert] = df_all[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_subset[cols_to_convert] = df_subset[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)

    for idx, row in df_subset.iterrows():
        male_counts = row[male_cols]
        female_counts = row[female_cols]
        total_population = male_counts.sum() + female_counts.sum()
        if total_population == 0:
            print(f"Skipping block {idx} with zero population")
            continue

        male_percent = (male_counts / total_population) * 100
        female_percent = (female_counts / total_population) * 100

        df_nonzero = df_all[df_all['P12_001N'] > 0]
        total_all = df_nonzero[male_cols + female_cols].sum().sum()
        male_avg = df_nonzero[male_cols].sum() / total_all * 100
        female_avg = df_nonzero[female_cols].sum() / total_all * 100

        # Symmetric x-axis range rounded up to the nearest 5% so the
        # ticks are clean across blocks.
        max_pct = max(male_percent.max(), female_percent.max(), male_avg.max(), female_avg.max())
        max_pct = np.ceil(max_pct / 5.0) * 5

        fig, ax = plt.subplots(figsize=(14, 9))

        ax.axvline(-center_pad, color='black', linewidth=2.5, zorder=5)
        ax.axvline(center_pad, color='black', linewidth=2.5, zorder=5)

        ax.barh(y, -male_percent.values, left=-center_pad,
                color='steelblue', alpha=0.7, label='Percent Male', zorder=3)
        ax.barh(y, female_percent.values, left=center_pad,
                color='lightcoral', alpha=0.7, label='Percent Female', zorder=3)

        ax.plot(-male_avg.values - center_pad, y, 'o', color='navy', label='NYC Avg Male', zorder=4)
        ax.plot(female_avg.values + center_pad, y, 'o', color='darkred', label='NYC Avg Female', zorder=4)

        for i, label in enumerate(age_labels):
            ax.text(0, i, label, ha='center', va='center', fontsize=label_fontsize,
                    fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9), zorder=2)

        ax.text(-center_pad - max_pct, len(age_labels) + 0.8, "Male",
                ha='center', fontsize=label_fontsize + 2, fontweight='bold', color='steelblue')
        ax.text(center_pad + max_pct, len(age_labels) + 0.8, "Female",
                ha='center', fontsize=label_fontsize + 2, fontweight='bold', color='darkred')

        ax.set_xlim(-center_pad - max_pct, center_pad + max_pct)
        ax.set_ylim(-1, len(age_labels))
        ax.set_yticks([])

        tick_vals = np.linspace(0, max_pct, 6)
        ticks_left = -tick_vals - center_pad
        ticks_right = tick_vals + center_pad
        ax.set_xticks(np.concatenate((ticks_left[::-1], [0], ticks_right)))
        ax.set_xticklabels(
            [f"{int(t)}%" for t in tick_vals[::-1]] + [""] + [f"{int(t)}%" for t in tick_vals],
            fontsize=label_fontsize
        )

        ax.set_xlabel("Percentage of Total Population (%)", fontsize=label_fontsize + 2)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')

        # Last four chars of the GEO_ID = block-within-block-group code.
        block_code = str(row['GEO_ID'])[-4:]
        ax.set_title(f"Age/Sex Population Distribution for Block {block_code}",
                     fontsize=label_fontsize + 4, fontweight='bold')

        ax.legend(loc='upper right', fontsize=label_fontsize)
        plt.tight_layout()
        plt.show()
        fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\population pyramid\\hb_{block_code}_popPyramid.png", dpi=800, bbox_inches='tight')


plot_population_pyramid_by_block(df_filtered, df_merged)


def plot_race_distribution_by_block(df_subset, df_all):
    """Plot the race / ethnicity distribution for the target blocks.

    Renders one aggregate plot for the union of the target blocks and
    one plot per individual block. Categories are sorted by NYC-wide
    descending percentage so the order is consistent across panels.
    Hispanic / Latino is treated as a single combined category that
    cuts across race.

    Parameters
    ----------
    df_subset : pandas.DataFrame
        DHC rows for the target census blocks.
    df_all : pandas.DataFrame
        DHC rows for all NYC census blocks.
    """
    # Selected race categories (non-Hispanic alone).
    race_cols = {
        "White alone, Non-Hispanic": "P5_003N",
        "Black or African American alone, Non-Hispanic": "P5_004N",
        "Asian alone, Non-Hispanic": "P5_006N",
    }
    # P5_011N..P5_017N covers Hispanic-or-Latino respondents across
    # multiple race classifications; we sum across them for a single
    # combined "Hispanic or Latino (All)" category.
    hispanic_cols = [f'P5_{i:03}N' for i in range(11, 18)]
    all_cols = list(race_cols.values()) + hispanic_cols + ['P5_001N']

    df_subset[all_cols] = df_subset[all_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_all[all_cols] = df_all[all_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    total_all_pop = df_all["P5_001N"].sum()
    race_vals_all = {label: df_all[col].sum() for label, col in race_cols.items()}
    race_vals_all["Hispanic or Latino (All)"] = df_all[hispanic_cols].sum().sum()
    avg_pct = {k: (v / total_all_pop) * 100 for k, v in race_vals_all.items()}

    # Sort categories by NYC-wide percentage so the largest groups
    # appear first.
    sorted_items = sorted(avg_pct.items(), key=lambda x: x[1], reverse=True)
    categories = [k for k, _ in sorted_items]

    # ---- Aggregate plot across all target blocks ----
    subset_total = df_subset[race_cols.values()].sum()
    subset_hispanic_total = df_subset[hispanic_cols].sum().sum()
    subset_pop_total = df_subset["P5_001N"].sum()

    if subset_pop_total > 0:
        combined_vals = {label: subset_total[race_cols[label]] for label in race_cols}
        combined_vals["Hispanic or Latino (All)"] = subset_hispanic_total
        combined_pct = {k: (v / subset_pop_total) * 100 for k, v in combined_vals.items()}

        print(f"\n--- Combined Race/Ethnicity Distribution for Hamilton/Howard Beach Census Blocks ---")
        for label in categories:
            count = int(combined_vals[label])
            print(f"  {label}: {count}")
        print(f"Total population: {int(subset_pop_total)}")

        values = [combined_pct[k] for k in categories]
        averages = [avg_pct[k] for k in categories]
        y = np.arange(len(categories))

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.barh(y, values, color='steelblue', alpha=0.85, label='Hamilton/Howard Beach Census Blocks Combined')
        ax.plot(averages, y, 'ko', label='NYC Census Block Avg')

        ax.set_yticks(y)
        ax.set_yticklabels(categories, fontsize=16)
        ax.invert_yaxis()
        ax.set_xlabel('Percentage of Total Population (%)', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.set_title('Race/Ethnicity Distribution (Hamilton/Howard Beach Census Blocks)', fontsize=20, fontweight='bold')
        ax.legend(loc='lower right', fontsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\allBlocks\\hb_allBlocks_raceDistribution.png", dpi=800, bbox_inches='tight')

    # ---- Per-block plots ----
    for idx, row in df_subset.iterrows():
        geo_id = str(row.get('GEO_ID', f'{idx}'))
        block_name = geo_id[-4:]
        total_pop = row["P5_001N"]
        if total_pop == 0:
            print(f"Skipping {block_name} due to zero population.")
            continue

        race_vals = {label: row[race_cols[label]] for label in race_cols}
        race_vals["Hispanic or Latino (All)"] = row[hispanic_cols].sum()
        race_pct = {k: (v / total_pop) * 100 for k, v in race_vals.items()}

        print(f"\n--- Race/Ethnicity Distribution for {block_name} ---")
        for label in categories:
            count = int(race_vals[label])
            print(f"  {label}: {count}")
        print(f"Total population: {int(total_pop)}")

        values = [race_pct[k] for k in categories]
        averages = [avg_pct[k] for k in categories]
        y = np.arange(len(categories))

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.barh(y, values, color='steelblue', alpha=0.85, label=block_name)
        ax.plot(averages, y, 'ko', label='NYC Census Block Avg')

        ax.set_yticks(y)
        ax.set_yticklabels(categories, fontsize=16)
        ax.invert_yaxis()
        ax.set_xlabel('Percentage of Total Population', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.set_title(f'Race/Ethnicity Distribution for {block_name}', fontsize=20, fontweight='bold')
        ax.legend(loc='lower right', fontsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\race distribution\\hb_{block_name}_raceDistribution.png", dpi=800, bbox_inches='tight')


plot_race_distribution_by_block(df_filtered, df_merged)


def plot_household_size_distribution(df_subset, df_all):
    """Plot household-size distribution (1, 2, 3, 4, 5+ persons).

    The raw H9 table breaks 5+-person households into three buckets
    (5, 6, 7+). For readability, we collapse these into a single 5+
    category. One aggregate plot is produced across target blocks plus
    one per individual block.

    Parameters
    ----------
    df_subset : pandas.DataFrame
        DHC rows for the target census blocks.
    df_all : pandas.DataFrame
        DHC rows for all NYC census blocks.
    """
    # Raw H9 size columns (H9_002N..H9_008N: 1-person through 7+).
    original_cols = [f'H9_{i:03}N' for i in range(2, 9)]
    total_col = 'H9_001N'
    cols_to_convert = original_cols + [total_col]

    df_all[cols_to_convert] = df_all[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_subset[cols_to_convert] = df_subset[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Collapse 5/6/7+ into a single 5+ category in both dataframes.
    for df in [df_all, df_subset]:
        df['H9_006N'] = df[['H9_006N', 'H9_007N', 'H9_008N']].sum(axis=1)

    size_cols = [f'H9_{i:03}N' for i in range(2, 7)]
    labels = [
        '1-person household',
        '2-person household',
        '3-person household',
        '4-person household',
        '5-or-more-person household'
    ]

    total_all = df_all[total_col].sum()
    all_counts = df_all[size_cols].sum()
    pct_all = (all_counts / total_all * 100).values

    subset_total = df_subset[total_col].sum()
    subset_counts = df_subset[size_cols].sum()
    subset_pct = (subset_counts / subset_total * 100).values

    # Aggregate across target blocks.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, subset_pct, color='seagreen', label='Hamilton/Howard Beach Census Blocks Total')
    ax.plot(pct_all, labels, 'ko', label='NYC Census Block Avg')

    ax.set_xlabel('Percentage of Households', fontsize=12)
    ax.set_title('Household Size Distribution (1-5+ persons)\nHamilton/Howard Beach Census Blocks', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\allBlocks\\hb_allBlocks_householdsizeDistribution.png", dpi=800, bbox_inches='tight')

    # Per-block.
    for _, row in df_subset.iterrows():
        geo_id = str(row['GEO_ID'])
        block_id = geo_id[-4:]
        block_counts = row[size_cols].astype(float)
        total = row[total_col]

        if total == 0:
            continue

        pct = (block_counts / total * 100).values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels, pct, color='mediumseagreen', label=f'Block {block_id}')
        ax.plot(pct_all, labels, 'ko', label='NYC Census Block Avg')

        ax.set_xlabel('Percentage of Households', fontsize=12)
        ax.set_title(f'Household Size Distribution (1-5+ persons)\nBlock {block_id}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.show()
    fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\race distribution\\hb_{block_id}_householdsizeDistribution.png", dpi=800, bbox_inches='tight')


plot_household_size_distribution(df_filtered, df_merged)


def plot_occupancy_distribution(df_subset, df_all):
    """Plot housing occupancy (occupied vs. vacant) for the target blocks.

    Aggregate plot across target blocks plus one plot per individual
    block, with NYC-wide averages overlaid.

    Parameters
    ----------
    df_subset : pandas.DataFrame
        DHC rows for the target census blocks.
    df_all : pandas.DataFrame
        DHC rows for all NYC census blocks.
    """
    # H3_001N: total housing units. H3_002N occupied, H3_003N vacant.
    total_col = 'H3_001N'
    tenure_cols = ['H3_002N', 'H3_003N']
    labels = ['Occupied', 'Vacant']

    df_subset[[total_col] + tenure_cols] = df_subset[[total_col] + tenure_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_all[[total_col] + tenure_cols] = df_all[[total_col] + tenure_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    total_all = df_all[total_col].sum()
    tenure_total_all = df_all[tenure_cols].sum()
    avg_pct = (tenure_total_all / total_all * 100).values

    total_subset = df_subset[total_col].sum()
    if total_subset > 0:
        tenure_subset_sum = df_subset[tenure_cols].sum()
        subset_pct = (tenure_subset_sum / total_subset * 100).values

        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.arange(len(labels))

        ax.barh(y, subset_pct, color='seagreen', edgecolor='black', label='Hamilton/Howard Beach Census Blocks Total')
        ax.plot(avg_pct, y, 'ko', label='NYC Census Block Avg')

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Percentage of Housing Units', fontsize=12)
        ax.set_title('Housing Occupancy Distribution\nHamilton/Howard Beach Census Blocks', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\allBlocks\\hb_allBlocks_occupancyDistribution.png", dpi=800, bbox_inches='tight')

    # Per-block.
    for _, row in df_subset.iterrows():
        total = row[total_col]
        if total == 0:
            continue

        geo_id = str(row['GEO_ID'])
        block_id = geo_id[-4:]
        values = row[tenure_cols].values.astype(float)
        pct = (values / total * 100)

        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.arange(len(labels))

        ax.barh(y, pct, color='cornflowerblue', edgecolor='black', label=f'Block {block_id}')
        ax.plot(avg_pct, y, 'ko', label='NYC Census Block Avg')

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Percentage of Housing Units', fontsize=12)
        ax.set_title(f'Housing Occupancy Distribution\nBlock {block_id}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\occupancy\\hb_{block_id}_occupancyDistribution.png", dpi=800, bbox_inches='tight')


plot_occupancy_distribution(df_filtered, df_merged)


def plot_housing_tenure_distribution(df_subset, df_all):
    """Plot housing tenure (owner-occupied vs. renter-occupied).

    H4_002N + H4_003N together cover owner-occupied units (mortgaged
    plus owned-free-and-clear); H4_004N covers renter-occupied units.
    The two owner sub-categories are summed for a binary owner / renter
    split.

    Parameters
    ----------
    df_subset : pandas.DataFrame
        DHC rows for the target census blocks.
    df_all : pandas.DataFrame
        DHC rows for all NYC census blocks.
    """
    total_col = 'H4_001N'
    owner_cols = ['H4_002N', 'H4_003N']
    renter_col = 'H4_004N'
    tenure_cols = owner_cols + [renter_col]
    labels = ['Owner-occupied', 'Renter-occupied']

    df_subset[[total_col] + tenure_cols] = df_subset[[total_col] + tenure_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_all[[total_col] + tenure_cols] = df_all[[total_col] + tenure_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    owner_total_all = df_all[owner_cols].sum().sum()
    renter_total_all = df_all[renter_col].sum()
    total_all = df_all[total_col].sum()
    avg_pct = np.array([owner_total_all, renter_total_all]) / total_all * 100

    # Aggregate across target blocks.
    owner_total_subset = df_subset[owner_cols].sum().sum()
    renter_total_subset = df_subset[renter_col].sum()
    total_subset = df_subset[total_col].sum()

    print(f"--- Combined Housing Tenure for Selected Census Blocks ---")
    print(f"  Owner-occupied: {int(owner_total_subset)}")
    print(f"  Renter-occupied: {int(renter_total_subset)}")
    print(f"Total housing units: {int(total_subset)}\n")

    if total_subset > 0:
        subset_pct = np.array([owner_total_subset, renter_total_subset]) / total_subset * 100

        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.arange(len(labels))

        ax.barh(y, subset_pct, color='seagreen', edgecolor='black', label='Selected Blocks Total')
        ax.plot(avg_pct, y, 'ko', label='NYC Census Block Avg')

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Percentage of Housing Units', fontsize=12)
        ax.set_title('Housing Tenure Distribution\nSelected NYC Census Blocks', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\allBlocks\\hb_allBlocks_housingtenureDistribution.png", dpi=800, bbox_inches='tight')

    # Per-block.
    for _, row in df_subset.iterrows():
        total = row[total_col]
        if total == 0:
            continue

        geo_id = str(row['GEO_ID'])
        block_id = geo_id[-4:]
        owner_count = row[owner_cols].sum()
        renter_count = row[renter_col]
        pct = np.array([owner_count, renter_count]) / total * 100

        print(f"--- Housing Tenure for {block_id} ---")
        print(f"  Owner-occupied: {int(owner_count)}")
        print(f"  Renter-occupied: {int(renter_count)}")
        print(f"Total housing units: {int(total)}\n")

        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.arange(len(labels))

        ax.barh(y, pct, color='cornflowerblue', edgecolor='black', label=f'Block {block_id}')
        ax.plot(avg_pct, y, 'ko', label='NYC Census Block Avg')

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Percentage of Housing Units', fontsize=12)
        ax.set_title(f'Housing Tenure Distribution\nBlock {block_id}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(f"F:\\dsl_CLIMA\\projects\\submittable\\clima\\plots\\demographics\\bar charts\\tenure\\hb_{block_id}_housingtenureDistribution.png", dpi=800, bbox_inches='tight')


plot_housing_tenure_distribution(df_filtered, df_merged)
