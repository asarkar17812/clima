"""
data_cleaning.py
================
Build the four export dataframes consumed by the rest of the project.

Outputs (written to ``export/``):

- ``df_outer_county.csv``: one row per cleaned FIPS county. Columns
  include within-county, cross-county, and total connection counts; the
  ESRI user-count and population estimates; the derived coverage
  estimate; CBSA metadata (code, title, Metropolitan/Micropolitan
  classification); and the rescaled / normalized variables used in the
  log-log regressions.
- ``df_cbsa.csv``: one row per CBSA. Columns include the three
  CBSA-level connection types (ICIC, ICCC, OCOC), aggregated user and
  population estimates, coverage, and rescaled / normalized variants.
- ``df_msa.csv``: subset of ``df_cbsa`` for Metropolitan Statistical
  Areas only, with normalization averages recomputed *within* the MSA
  subpopulation so each fit is internally consistent.
- ``df_musa.csv``: subset of ``df_cbsa`` for Micropolitan Statistical
  Areas only, with normalization averages recomputed within the muSA
  subpopulation.

Inputs (read from ``source/``):

- ``source/sci/county_county.tsv``: Meta Social Connectedness Index,
  October 2021, county-to-county. Columns: ``user_loc``, ``fr_loc``,
  ``scaled_sci``. Includes both (i,j) and (j,i) entries.
- ``source/users/county_users.csv``: ESRI 2022 county estimates.
  ``MP19049a_B`` is the 2022 Facebook MAU estimate, ``TOTPOP_CY`` is the
  2022 total population, ``ID`` is the 5-digit FIPS county code.
- ``source/crosswalk/list1.xls``: U.S. BLS county-to-CBSA crosswalk used
  to map each FIPS county to its CBSA, MSA, or muSA.

Method:
Connection counts are recovered from the Meta SCI formula by inverting
the published normalization. CBSA-level totals exploit the conservation
identity ``Total = ICIC + ICCC + OCOC`` so the cross-covering OCOC term
is recovered by subtraction rather than enumerated over the full ~10M-row
symmetric edge list explicitly. Coverage rescaling and normalization by
the GEOID-level averages follow Schlapfer et al. (2014).
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Load raw inputs.
# All identifiers are read as strings so that leading zeros on FIPS codes
# (e.g. "01001" for Autauga County, AL) are preserved through joins.
# ----------------------------------------------------------------------
df_sci = pd.read_table('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\sci\\county_county.tsv', dtype=str)
df_users = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\users\\county_users.csv', dtype=str)
df_crosswalk = pd.read_excel('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\crosswalk\\list1.xls', header=2, dtype=str)

# ----------------------------------------------------------------------
# BLS crosswalk: drop the three trailing footer rows, combine the state
# and county FIPS columns into one 5-digit FIPS, and keep only the
# columns we need downstream.
# ----------------------------------------------------------------------
df_crosswalk = df_crosswalk.iloc[:-3]
df_crosswalk['FIPS'] = df_crosswalk['FIPS State Code'] + df_crosswalk['FIPS County Code']
df_crosswalk['FIPS'] = df_crosswalk['FIPS'].astype(str).str.zfill(5)

# Connecticut reorganization (2023): the Census Bureau replaced
# Connecticut's eight historical counties with nine Council-of-
# Governments planning regions as the new "county-equivalent" units.
# The October 2021 SCI and the 2022 ESRI user/population estimates
# both predate this transition and use the original eight-county FIPS
# codes (09001 Fairfield through 09015 Windham). When the BLS crosswalk
# vintage carries the new planning-region FIPS (09110..09190) instead,
# we remap each new code back to the historical county it overlaps
# most heavily with so the crosswalk joins cleanly against the SCI and
# ESRI tables. The mapping follows the dominant geographic overlap
# reported in the official Census Bureau crosswalk between the two
# vintages; the two planning regions that straddle more than one
# historical county (Naugatuck Valley and Western Connecticut) are
# assigned to the county containing the larger share of their 2020
# population. After the remap we drop the duplicate rows that the
# many-to-one collapse creates (e.g., both 09110 and the original 09003
# Hartford mapping to 09003), keeping the first crosswalk entry for
# each FIPS. This is the mirror image of the Alaska reorganization
# below: Alaska had two areas merge into one, Connecticut had eight
# areas split into nine, and in both cases we collapse the post-
# transition vintage back to the pre-transition FIPS so the rest of
# the pipeline sees a single consistent geography.
ct_planning_to_county = {
    '09110': '09003',  # Capitol Region -> Hartford
    '09120': '09001',  # Greater Bridgeport -> Fairfield
    '09130': '09007',  # Lower CT River Valley -> Middlesex
    '09140': '09009',  # Naugatuck Valley -> New Haven (dominant overlap)
    '09150': '09015',  # Northeastern CT -> Windham
    '09160': '09005',  # Northwest Hills -> Litchfield
    '09170': '09009',  # South Central CT -> New Haven
    '09180': '09011',  # Southeastern CT -> New London
    '09190': '09001',  # Western CT -> Fairfield (dominant overlap)
}
df_crosswalk['FIPS'] = df_crosswalk['FIPS'].replace(ct_planning_to_county)
df_crosswalk = df_crosswalk.drop_duplicates(subset=['FIPS'], keep='first')

df_crosswalk = df_crosswalk[['FIPS', 'CBSA Code', 'Metropolitan/Micropolitan Statistical Area', 'CBSA Title']].copy()

# ----------------------------------------------------------------------
# ESRI users: keep only the MAU estimate, population, and FIPS columns;
# coerce to numeric; aggregate any duplicate FIPS rows by sum.
# ----------------------------------------------------------------------
df_users = df_users[['MP19049a_B', 'TOTPOP_CY', 'ID']]
df_users = df_users.rename(columns={'MP19049a_B': 'user_count', 'TOTPOP_CY': 'pop', 'ID': 'FIPS'})

df_users['FIPS'] = df_users['FIPS'].astype(str).str.zfill(5)
df_users['user_count'] = pd.to_numeric(df_users['user_count'], errors='coerce')
df_users['pop'] = pd.to_numeric(df_users['pop'], errors='coerce')

df_users = df_users.set_index('FIPS')
df_users = df_users.groupby(df_users.index)[['user_count', 'pop']].sum()

# Alaska reorganization: 02261 (Valdez-Cordova) was split into 02063
# (Chugach) and 02066 (Copper River) after the 2020 redistricting. We
# re-merge them under 02261 so the SCI (which still uses the pre-split
# replacement code) joins cleanly to ESRI.
# 15005 (Kalawao County, HI) is dropped because its population is in the
# low double digits and produces extreme outliers on log-scale plots.
df_users.loc['02261'] = df_users.loc[['02066', '02063']].sum()
df_users = df_users.drop(index=['02066', '02063', '15005'])

# ----------------------------------------------------------------------
# Restrict the SCI to cleaned FIPS pairs and recover Facebook connection
# counts. The published SCI is multiplied by 1e9, hence the division.
#
# Meta SCI formulas (inverted to solve for connection counts):
#   SCI_{i,i} = FB_Conn_{i,i} / (|S_i| * (|S_i| - 1))
#   SCI_{i,j} = FB_Conn_{i,j} / (|S_i| * |S_j|)
# ----------------------------------------------------------------------
df_sci = df_sci[df_sci['user_loc'].isin(df_users.index) & df_sci['fr_loc'].isin(df_users.index)]
df_sci['scaled_sci'] = df_sci['scaled_sci'].astype(float)
df_sci['scaled_sci'] = df_sci['scaled_sci'] / 1000000000

df_sci['user_user_count'] = df_sci['user_loc'].map(df_users['user_count'])
df_sci['user_pop'] = df_sci['user_loc'].map(df_users['pop'])
df_sci['fr_user_count'] = df_sci['fr_loc'].map(df_users['user_count'])
df_sci['fr_pop'] = df_sci['fr_loc'].map(df_users['pop'])

# Within-county rows use the homogeneous SCI denominator; cross-county
# rows use the heterogeneous denominator.
df_sci['Connections'] = np.where(
    df_sci['user_loc'] == df_sci['fr_loc'],
    df_sci['scaled_sci'] * (df_sci['user_user_count'] * (df_sci['user_user_count'] - 1)),
    df_sci['scaled_sci'] * df_sci['user_user_count'] * df_sci['fr_user_count']
)

# Attach CBSA metadata for both endpoints of each SCI row so we can group
# by CBSA later. Suffix '_fr' is the partner county.
df_county = df_sci.merge(df_crosswalk, left_on='user_loc', right_on='FIPS', how='left', suffixes=('', '_user')).copy()
df_county = df_county.merge(df_crosswalk, left_on='fr_loc', right_on='FIPS', how='left', suffixes=('', '_fr')).copy()

df_county['user_user_count'] = pd.to_numeric(df_county['user_user_count'], errors='coerce')
df_county['user_pop'] = pd.to_numeric(df_county['user_pop'], errors='coerce')

# ----------------------------------------------------------------------
# County-level aggregation.
# - df_inner_county: one row per home county with the within-county
#   connection count plus user/pop/CBSA metadata.
# - df_outer_county: sum of all cross-county connections originating in
#   the home county.
# Inner + Outer = Total county-level degree.
# ----------------------------------------------------------------------
df_inner_county = (
    df_county[df_county['user_loc'] == df_county['fr_loc']]
    .groupby('user_loc', as_index=False)
    .agg(
        user_est=('user_user_count', 'first'),
        pop_est=('user_pop', 'first'),
        metro_micro_area=('Metropolitan/Micropolitan Statistical Area', 'first'),
        CBSA_code=('CBSA Code', 'first'),
        CBSA_title=('CBSA Title', 'first'),
        inter_county_connections=('Connections', 'first'),
    )
).copy()

df_outer_county = (
    df_county[df_county['user_loc'] != df_county['fr_loc']]
    .groupby('user_loc', as_index=False)
    .agg(outer_county_connections=('Connections', 'sum'))
).copy()

df_inner_county['outer_county_connections'] = df_outer_county['outer_county_connections']
df_inner_county['total connections'] = (
    df_outer_county['outer_county_connections'] + df_inner_county['inter_county_connections']
)

# ----------------------------------------------------------------------
# CBSA-level aggregation.
#
# ICIC: within-CBSA, within-county. Sum of (i,i) connections for i in C.
# ICCC: within-CBSA, between-county. Sum of (i,j) with i != j and both
#       in the same CBSA.
# OCOC: outside-CBSA. (i,j) with i in C and j outside C.
#
# Note: when a CBSA contains exactly one county, ICCC is undefined / 0,
# which is handled by the np.where on the ``total inter_cbsa`` column.
# ----------------------------------------------------------------------
df_inter_county_inter_cbsa = (
    df_county[
        (df_county['CBSA Code'] == df_county['CBSA Code_fr']) &
        (df_county['user_loc'] == df_county['fr_loc'])
    ]
    .groupby('CBSA Code', as_index=False)
    .agg(
        CBSA_title=('CBSA Title', 'first'),
        metro_micro_area=('Metropolitan/Micropolitan Statistical Area', 'first'),
        user_est=('user_user_count', 'sum'),
        pop_est=('user_pop', 'sum'),
        inter_cbsa_connections=('Connections', 'sum'),
    )
).copy()

df_outer_county_inter_cbsa = (
    df_county[df_county['CBSA Code'] == df_county['CBSA Code_fr']]
    .query("user_loc != fr_loc")
    .groupby('CBSA Code', as_index=False)
    .agg(outer_county_inter_cbsa_connections=('Connections', 'sum'))
).copy()

df_outer_cbsa = (
    df_county[
        (df_county['CBSA Title'] != df_county['CBSA Title_fr']) &
        (df_county['user_loc'] != df_county['fr_loc'])
    ]
    .groupby('CBSA Code', as_index=False)
    .agg(outer_cbsa_connections=('Connections', 'sum'))
).copy()

df_cbsa = (
    df_inter_county_inter_cbsa
    .merge(df_outer_county_inter_cbsa, on='CBSA Code', how='left')
    .merge(df_outer_cbsa, on='CBSA Code', how='left')
)

# Single-county CBSAs have no ICCC term, so the inter-CBSA total is
# just ICIC. Multi-county CBSAs add the ICCC piece on top.
df_cbsa['total inter_cbsa connections'] = np.where(
    df_county.groupby('CBSA Code')['user_loc'].nunique() == 1,
    df_cbsa['inter_cbsa_connections'],
    df_cbsa['inter_cbsa_connections'] + df_cbsa['outer_county_inter_cbsa_connections']
)

df_cbsa['total connections'] = df_cbsa['total inter_cbsa connections'] + df_cbsa['outer_cbsa_connections']

# ----------------------------------------------------------------------
# Coverage rescaling (Schlapfer et al. 2014).
# s = |S| / N. Rescaled degree K_r = k / s recovers the population-level
# degree the SCI undercounts because it only sees the Facebook-active
# subset of residents.
# ----------------------------------------------------------------------
df_cbsa['coverage est'] = df_cbsa['user_est'] / df_cbsa['pop_est']

df_cbsa['rescaled total inter_cbsa connections'] = df_cbsa['total inter_cbsa connections'] / df_cbsa['coverage est']
df_cbsa['rescaled outer_cbsa_connections'] = df_cbsa['outer_cbsa_connections'] / df_cbsa['coverage est']
df_cbsa['rescaled total connections'] = df_cbsa['total connections'] / df_cbsa['coverage est']

# ----------------------------------------------------------------------
# Subset to MSA-only and muSA-only views. The normalization averages
# below are taken *within* each subpopulation so each regression's
# normalization is internally consistent and not contaminated by the
# size mismatch between metropolitan and micropolitan areas.
# ----------------------------------------------------------------------
df_msa = df_cbsa[df_cbsa['metro_micro_area'] == 'Metropolitan Statistical Area'].copy()
df_musa = df_cbsa[df_cbsa['metro_micro_area'] == 'Micropolitan Statistical Area'].copy()

df_msa['normed pop_est'] = df_msa['pop_est'] / df_msa['pop_est'].mean()
df_musa['normed pop_est'] = df_musa['pop_est'] / df_musa['pop_est'].mean()
df_cbsa['normed pop_est'] = df_cbsa['pop_est'] / df_cbsa['pop_est'].mean()

# Normalize each rescaled-connection column by its own mean so the
# resulting series is centered around 1 and the OLS fit becomes
# scale-invariant.
df_msa['rescaled total inter_cbsa connections'] = (
    df_msa['rescaled total inter_cbsa connections'] / df_msa['rescaled total inter_cbsa connections'].mean()
)
df_msa['rescaled outer_cbsa_connections'] = (
    df_msa['rescaled outer_cbsa_connections'] / df_msa['rescaled outer_cbsa_connections'].mean()
)
df_msa['rescaled total connections'] = (
    df_msa['rescaled total connections'] / df_msa['rescaled total connections'].mean()
)

df_musa['rescaled total inter_cbsa connections'] = (
    df_musa['rescaled total inter_cbsa connections'] / df_musa['rescaled total inter_cbsa connections'].mean()
)
df_musa['rescaled outer_cbsa_connections'] = (
    df_musa['rescaled outer_cbsa_connections'] / df_musa['rescaled outer_cbsa_connections'].mean()
)
df_musa['rescaled total connections'] = (
    df_musa['rescaled total connections'] / df_musa['rescaled total connections'].mean()
)

df_cbsa['rescaled total inter_cbsa connections'] = (
    df_cbsa['rescaled total inter_cbsa connections'] / df_cbsa['rescaled total inter_cbsa connections'].mean()
)
df_cbsa['rescaled outer_cbsa_connections'] = (
    df_cbsa['rescaled outer_cbsa_connections'] / df_cbsa['rescaled outer_cbsa_connections'].mean()
)
df_cbsa['rescaled total connections'] = (
    df_cbsa['rescaled total connections'] / df_cbsa['rescaled total connections'].mean()
)

# County-level coverage rescaling and normalization (parallel to the
# CBSA-level operations above).
df_inner_county['coverage est'] = df_inner_county['user_est'] / df_inner_county['pop_est']
df_inner_county['normed pop_est'] = df_inner_county['pop_est'] / df_inner_county['pop_est'].mean()

df_inner_county['rescaled inter_county_connections'] = (
    df_inner_county['inter_county_connections'] / df_inner_county['coverage est']
)
df_inner_county['rescaled outer_county_connections'] = (
    df_inner_county['outer_county_connections'] / df_inner_county['coverage est']
)
df_inner_county['rescaled total connections'] = (
    df_inner_county['total connections'] / df_inner_county['coverage est']
)

df_inner_county['rescaled inter_county_connections'] = (
    df_inner_county['rescaled inter_county_connections'] / df_inner_county['rescaled inter_county_connections'].mean()
)
df_inner_county['rescaled outer_county_connections'] = (
    df_inner_county['rescaled outer_county_connections'] / df_inner_county['rescaled outer_county_connections'].mean()
)
df_inner_county['rescaled total connections'] = (
    df_inner_county['rescaled total connections'] / df_inner_county['rescaled total connections'].mean()
)

# ----------------------------------------------------------------------
# Export the four working dataframes for downstream visualization
# scripts and the notebook.
# ----------------------------------------------------------------------
df_inner_county.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_outer_county.csv', index=False)
df_cbsa.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_cbsa.csv', index=False)
df_msa.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_msa.csv', index=False)
df_musa.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_musa.csv', index=False)
