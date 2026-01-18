import pandas as pd 
import numpy as np 

df_sci = pd.read_table('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\sci\\county_county.tsv', dtype=str)
df_users = pd.read_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\users\\county_users.csv', dtype=str)
df_crosswalk = pd.read_excel('F:\\dsl_CLIMA\\projects\\submittable\\clima\\source\\crosswalk\\list1.xls', header=2, dtype=str)

df_crosswalk = df_crosswalk.iloc[:-3]
df_crosswalk['FIPS'] = df_crosswalk['FIPS State Code'] + df_crosswalk['FIPS County Code']
df_crosswalk['FIPS'] = df_crosswalk['FIPS'].astype(str).str.zfill(5)
df_crosswalk = df_crosswalk[['FIPS','CBSA Code', 'Metropolitan/Micropolitan Statistical Area', 'CBSA Title']].copy()

df_users = df_users[['MP19049a_B', 'TOTPOP_CY', 'ID']]
df_users = df_users.rename(columns={'MP19049a_B': 'user_count','TOTPOP_CY': 'pop','ID': 'FIPS'})

df_users['FIPS'] = df_users['FIPS'].astype(str).str.zfill(5)
df_users['user_count'] = pd.to_numeric(df_users['user_count'], errors='coerce')
df_users['pop']  = pd.to_numeric(df_users['pop'], errors='coerce')

df_users = df_users.set_index('FIPS')
df_users = df_users.groupby(df_users.index)[['user_count', 'pop']].sum()

# Chugach + Copperhead Census Areas = Valdez-Cordova Census Area (after 2020)
df_users.loc['02261'] = df_users.loc[['02066', '02063']].sum()
df_users = df_users.drop(index=['02066', '02063', '15005']) # 15005 is Kalawao County in Hawaii 

df_sci = df_sci[df_sci['user_loc'].isin(df_users.index) & df_sci['fr_loc'].isin(df_users.index)]
df_sci['scaled_sci'] = df_sci['scaled_sci'].astype(float)
df_sci['scaled_sci'] = df_sci['scaled_sci']/1000000000
df_sci['user_user_count'] = df_sci['user_loc'].map(df_users['user_count'])
df_sci['user_pop'] = df_sci['user_loc'].map(df_users['pop'])
df_sci['fr_user_count'] = df_sci['fr_loc'].map(df_users['user_count'])
df_sci['fr_pop'] = df_sci['fr_loc'].map(df_users['pop'])

df_sci['Connections'] = np.where(df_sci['user_loc'] == df_sci['fr_loc'], df_sci['scaled_sci'] * (df_sci['user_user_count'] * (df_sci['user_user_count'] - 1)), df_sci['scaled_sci'] * df_sci['user_user_count'] * df_sci['fr_user_count'])
df_county = df_sci.merge(df_crosswalk, left_on = 'user_loc', right_on = 'FIPS', how = 'left', suffixes=('', '_user')).copy()
df_county = df_county.merge(df_crosswalk, left_on = 'fr_loc', right_on = 'FIPS', how = 'left', suffixes=('', '_fr')).copy()

df_county['user_user_count'] = pd.to_numeric(df_county['user_user_count'], errors='coerce')
df_county['user_pop']  = pd.to_numeric(df_county['user_pop'], errors='coerce')

df_inner_county = (
    df_county[(df_county["user_loc"] == df_county["fr_loc"])]
    .groupby("user_loc", as_index=False)
    .agg(
        user_est=("user_user_count", "first"),
        pop_est=("user_pop", "first"),
        metro_micro_area=("Metropolitan/Micropolitan Statistical Area", "first"),
        CBSA_code=("CBSA Code", "first"),
        CBSA_title=("CBSA Title", "first"),
        inter_county_connections=("Connections", "first"),
    )
).copy()

df_outer_county = (
    df_county[(df_county["user_loc"] != df_county["fr_loc"])]
    .groupby("user_loc", as_index=False)
    .agg(outer_county_connections=("Connections", "sum"))
).copy()

df_inner_county['outer_county_connections'] = df_outer_county['outer_county_connections']
df_inner_county['total connections'] = df_outer_county['outer_county_connections'] + df_inner_county['inter_county_connections']

df_inter_county_inter_cbsa = (df_county[
        (df_county["CBSA Code"] == df_county["CBSA Code_fr"]) &
        (df_county["user_loc"] == df_county["fr_loc"])
    ]
    .groupby("CBSA Code", as_index=False)
    .agg(
        CBSA_title=("CBSA Title", "first"),
        metro_micro_area=("Metropolitan/Micropolitan Statistical Area", "first"),
        user_est=("user_user_count", "sum"),
        pop_est=("user_pop", "sum"),
        inter_cbsa_connections=("Connections", "sum"),
        )
).copy()

df_outer_county_inter_cbsa = (
    df_county[df_county["CBSA Code"] == df_county["CBSA Code_fr"]]
    .query("user_loc != fr_loc")
    .groupby("CBSA Code", as_index=False)
    .agg(outer_county_inter_cbsa_connections=("Connections", "sum"))
).copy()

df_outer_cbsa = (
       df_county[
        (df_county["CBSA Title"] != df_county["CBSA Title_fr"]) &
        (df_county["user_loc"] != df_county["fr_loc"])
    ]
    .groupby("CBSA Code", as_index=False)
    .agg(outer_cbsa_connections=("Connections", "sum"))
).copy()

df_cbsa = df_inter_county_inter_cbsa.merge(df_outer_county_inter_cbsa, on='CBSA Code', how='left').merge(df_outer_cbsa, on='CBSA Code', how='left')

df_cbsa['total inter_cbsa connections'] = np.where(
    df_county.groupby('CBSA Code')['user_loc'].nunique() == 1, 
    df_cbsa['inter_cbsa_connections'], 
    df_cbsa['inter_cbsa_connections'] + df_cbsa['outer_county_inter_cbsa_connections']
)

df_cbsa['total connections'] = df_cbsa['total inter_cbsa connections'] + df_cbsa['outer_cbsa_connections']
df_cbsa['coverage est'] = df_cbsa['user_est']/df_cbsa['pop_est']

df_cbsa['rescaled total inter_cbsa connections'] = df_cbsa['total inter_cbsa connections']/df_cbsa['coverage est']
df_cbsa['rescaled outer_cbsa_connections'] = df_cbsa['outer_cbsa_connections']/df_cbsa['coverage est']
df_cbsa['rescaled total connections'] = df_cbsa['total connections']/df_cbsa['coverage est']

df_msa = df_cbsa[df_cbsa['metro_micro_area'] == 'Metropolitan Statistical Area'].copy()
df_musa = df_cbsa[df_cbsa['metro_micro_area'] == 'Micropolitan Statistical Area'].copy()
df_msa['normed pop_est'] = (df_msa['pop_est']/(df_msa['pop_est'].mean()))
df_musa['normed pop_est'] = (df_musa['pop_est']/(df_musa['pop_est'].mean()))
df_cbsa['normed pop_est'] = (df_cbsa['pop_est']/(df_cbsa['pop_est'].mean()))

df_msa['rescaled total inter_cbsa connections'] = df_msa['rescaled total inter_cbsa connections']/df_msa['rescaled total inter_cbsa connections'].mean()
df_msa['rescaled outer_cbsa_connections'] = df_msa['rescaled outer_cbsa_connections']/df_msa['rescaled outer_cbsa_connections'].mean()
df_msa['rescaled total connections'] = df_msa['rescaled total connections']/df_msa['rescaled total connections'].mean()

df_musa['rescaled total inter_cbsa connections'] = df_musa['rescaled total inter_cbsa connections']/df_musa['rescaled total inter_cbsa connections'].mean()
df_musa['rescaled outer_cbsa_connections'] = df_musa['rescaled outer_cbsa_connections']/df_musa['rescaled outer_cbsa_connections'].mean()
df_musa['rescaled total connections'] = df_musa['rescaled total connections']/df_musa['rescaled total connections'].mean()

df_cbsa['rescaled total inter_cbsa connections'] = df_cbsa['rescaled total inter_cbsa connections']/df_cbsa['rescaled total inter_cbsa connections'].mean()
df_cbsa['rescaled outer_cbsa_connections'] = df_cbsa['rescaled outer_cbsa_connections']/df_cbsa['rescaled outer_cbsa_connections'].mean()
df_cbsa['rescaled total connections'] = df_cbsa['rescaled total connections']/df_cbsa['rescaled total connections'].mean()

df_inner_county['coverage est'] = df_inner_county['user_est']/df_inner_county['pop_est']
df_inner_county['normed pop_est'] = (df_inner_county['pop_est']/(df_inner_county['pop_est'].mean()))

df_inner_county['rescaled inter_county_connections'] = df_inner_county['inter_county_connections']/df_inner_county['coverage est']
df_inner_county['rescaled outer_county_connections'] = df_inner_county['outer_county_connections']/df_inner_county['coverage est']
df_inner_county['rescaled total connections'] = df_inner_county['total connections']/df_inner_county['coverage est']

df_inner_county['rescaled inter_county_connections'] = df_inner_county['rescaled inter_county_connections']/df_inner_county['rescaled inter_county_connections'].mean()
df_inner_county['rescaled outer_county_connections'] = df_inner_county['rescaled outer_county_connections']/df_inner_county['rescaled outer_county_connections'].mean()
df_inner_county['rescaled total connections'] = df_inner_county['rescaled total connections']/df_inner_county['rescaled total connections'].mean()

df_inner_county.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_outer_county.csv', index=False)
df_cbsa.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_cbsa.csv', index=False)
df_msa.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_msa.csv', index=False)
df_musa.to_csv('F:\\dsl_CLIMA\\projects\\submittable\\clima\\export\\df_musa.csv', index=False)