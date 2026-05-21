# Scaling of Social Connections & Demographic Analysis

## Ayush Sarkar: 5/15/2025 - 7/11/2025 | CLIMA w/ Dynamical Systems Lab @ NYU

*A study of how Facebook-derived social connectivity scales with population across U.S. counties and Core-Based Statistical Areas (CBSAs), paired with Census-block demographic profiles and homeowner-interview ground-truthing for coastal NYC communities under flood risk.*

---

![asarkar_nyu_ugsrp_poster](https://github.com/user-attachments/assets/de930506-2524-41f4-acb0-c01ba97a09e1)

> *Conference poster (also available in this repo as [`clima_poster.pdf`](clima_poster.pdf)). The full annotated analysis lives in [`clima.ipynb`](clima.ipynb); a high-resolution PDF of every figure produced is available at [`clima_figures.pdf`](clima_figures.pdf).*

---

## TL;DR

I take Meta's county-to-county Social Connectedness Index (SCI) together with ESRI 2022 Facebook MAU and population estimates and recover the absolute number of Facebook connections at the county, CBSA, MSA, and muSA levels. I then fit log-log power-law regressions of the form $k = N^{\beta} \cdot \epsilon$ — following the *Schläpfer et al. (2014)* coverage-rescaling methodology — to ask how social connectivity scales with population. The headline finding holds across every GEOID level: inner (within-place) connections are superlinear ($\beta \in [1.054, 1.082]$), outgoing (cross-place) connections are sublinear ($\beta \in [0.914, 0.945]$), and total connections are very close to but slightly below linear ($\beta \in [0.969, 0.982]$). On the demographic side, I produce Census-block-level analyses of Hamilton Beach / Howard Beach (Queens) and Red Hook (Brooklyn) — two NYC coastal communities sitting in FEMA flood zones — covering age/sex, race/ethnicity, household size, occupancy, tenure, educational attainment, and household income. Those quantitative summaries are paired with qualitative interviews of resident homeowners that ground the modeling choices in lived experience.

For the full methodology, all derivations, and every figure produced, open [`clima.ipynb`](clima.ipynb). The standalone scripts under [`scripts/`](scripts/) mirror the notebook cells one-to-one.

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Glossary](#glossary)
3. [Data Sources](#data-sources)
4. [Methodology](#methodology)
5. [Results: Scaling](#results-scaling)
6. [Results: Geographic Structure (Choropleths)](#results-geographic-structure-choropleths)
7. [Results: Distributions](#results-distributions)
8. [Results: Demographic Profiles](#results-demographic-profiles)
9. [Grounding the Model: Demographic Summaries & Resident Interviews](#grounding-the-model-demographic-summaries--resident-interviews)
10. [Interpretation: Why This Matters](#interpretation-why-this-matters)
11. [Caveats](#caveats)
12. [Project Layout](#project-layout)
13. [References](#references)

---

## Background & Motivation

CiviL Infrastructure research for climate change Mitigation and Adaptation (CLIMA) is a research effort focused on infrastructure research to help develop equitable and feasible solutions to the increasingly urgent threats posed by climate change, specifically through the mitigation of damages and adaptation to hazards and changes across coastal communities worldwide.

This leg of the CLIMA project is an interdisciplinary research effort that aims to model the effects of flood risk on coastal communities through a detailed investigation of the social networks, alongside other factors, in an attempt to build a model more capable of capturing human mobility phenomena — especially amongst homeowners. By using a modified compartmental model, we partition the population into sets and utilize the mean-field hypothesis to treat individuals as identical, thus allowing us to focus on population-level dynamics instead of individual-level cognition. To better appreciate the unique perspectives of the individuals and communities that most directly face these threats, this project also includes qualitative information about the network extracted through interviews with homeowners of coastal communities here in NYC. Furthermore, demographic breakdowns of the sampled coastal communities are used to increase our understanding of the context, everyday lives, and equity issues faced by the homeowners within these communities.

Two strands of literature anchor the quantitative side of this notebook. The first is urban scaling — work in the tradition of *Bettencourt, West, Schläpfer et al.* that consistently finds the intensity of human interaction within a city scales **superlinearly** with population: bigger cities don't just have more people, they have more interaction *per person*, with the scaling exponent typically falling in the $\beta \approx 1.1\!-\!1.2$ range for European face-to-face and phone-call data. If that pattern holds for online social ties, then aggregate social-network connectivity is itself a population-driven exposure variable, with direct implications for how shocks — including flood events — propagate through homeowner-decision networks. The second is the mobility-and-migration tradition of *Simini et al. (2012)*, which treats opportunity ratios as the driver of long-range moves. For coastal homeowners under flood risk, the relevant "opportunity" becomes the availability of comparable homes nearby, weighted by social ties to the destination. The scaling exponents quantified in this notebook are the natural empirical input to that kind of model.

The demographic side then places this national-level scaling work in concrete neighborhood context — these are real, mostly homeowner, mostly owner-occupied communities sitting inside FEMA flood zones, and any model built on the scaling above eventually has to be reconciled with what those neighborhoods actually look like up close.

---

## Glossary

| Symbol                              | Meaning                                                                                       |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| $G$                                 | The full county-level social network graph                                                    |
| $C$                                 | A GEOID-level covering (e.g., the set of counties forming a single CBSA)                       |
| $\tilde{C}$                         | Complement of $C$ (all counties outside $C$)                                                  |
| $n_{\text{GEOID}}$                  | Number of GEOID-level nodes in $G$                                                            |
| $N_i$                               | Population of GEOID $i$ (ESRI 2022 estimate)                                                  |
| $\|S_i\|$                             | 2022 Facebook MAU estimate (30-day) for GEOID $i$                                              |
| $s_i = \|S_i\|/N_i$                   | Coverage estimate for GEOID $i$                                                               |
| $k_{i,\,ic}$                        | Inter-GEOID degree — within-GEOID connections, i.e., $(i,i)$ pairs                            |
| $k_{i,\,o}$                         | Outgoing degree — cross-GEOID connections, i.e., $(i,j)$ with $i \neq j$                       |
| $k_{i,\,t} = k_{i,\,ic} + k_{i,\,o}$ | Total degree of GEOID $i$                                                                     |
| $K_{r,\,i} = k_{i,\,t} / s_i$       | Coverage-rescaled cumulative degree                                                           |
| $\langle K_r \rangle$               | Average rescaled degree across all GEOIDs of a given type                                     |
| $\beta$                             | Scaling exponent — the slope of the log-log regression                                         |
| $\gamma$                            | Intercept of the log-log regression                                                            |
| **ICIC**                            | Inter-Covering Inter-County — within-CBSA, within-county connections                          |
| **ICCC**                            | Inter-Covering Cross-County — within-CBSA, between-county connections                         |
| **OCOC**                            | Outer-Covering Cross-County — connections from a CBSA's counties to counties outside the CBSA |

> **A note on terminology**: throughout this project, the prefix "**Inter-**" denotes *within-GEOID* connections, while "**Outer-**" / "**Outgoing**" denote *cross-GEOID* connections. This is non-standard — conventionally "inter-" means "between" — but it's used consistently across the codebase. If you read `inter_county_connections` in the export CSVs, that is the within-county degree.

---

## Data Sources

| Source                                            | Resolution                  | Role in this analysis                                                                                                  |
| ------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Meta Social Connectedness Index (SCI), Oct 2021** | County-to-county (directed) | Friendship-tie measure; the input to every connection-count calculation                                                |
| **ESRI 2022 MAU & Population Estimates**          | 5-digit FIPS county         | Per-county Facebook MAU $\|S_i\|$ (column `MP19049a_B`) and population $N_i$ (column `TOTPOP_CY`); together yield $s_i$ |
| **BLS County-to-CBSA Crosswalk**                  | County $\to$ CBSA / MSA / muSA | Maps each county to its CBSA covering and labels it metropolitan vs. micropolitan                                      |
| **Meta Q4 '22 Earnings (DAU / MAU disclosures)**  | North America aggregate     | External sanity check on $s_i$ — Meta reports ~199M DAU / ~266M MAU vs. ~372M NA population, so $s_{\text{NA}} \approx 0.54\!-\!0.71$ |
| **2020 Decennial Census DHC**                     | Census block                | H3 (occupancy), H4 (tenure), H9 (household size), P3 (race), P5 (Hispanic origin), P12 (sex by age)                    |
| **IPUMS NHGIS `nhgis0004_ds267_20235`** (ACS 5-yr) | Block group                 | Educational attainment and household income for the Hamilton Beach / Red Hook pie charts                               |
| **TIGER/Line shapefiles (2021)**                  | County and CBSA polygons    | Choropleth basemaps                                                                                                    |
| **Homeowner interviews (HB / RH)**                 | Individual                  | Semi-structured interview transcriptions covering flood-risk perception, mobility intentions, and community ties        |

After cleaning, the working dataset comprises **3,141 unique FIPS codes**, **917 CBSA codes** (381 MSA + 536 muSA), and **9,865,881 SCI row entries** (= $2 \binom{3{,}141}{2} + 3{,}141$).

A few non-obvious cleaning steps are worth flagging. The post-2020 split of Alaska's Valdez-Cordova Census Area into Chugach (`02063`) and Copper River (`02066`) is re-merged into the pre-split-era code `02261` so the SCI (which still uses that code) joins cleanly to the ESRI demographics. Kalawao County, HI (`15005`) is dropped because its population is in the low double digits and breaks log-scale plots. And the directed SCI dataset is kept symmetric throughout — naive unsymmetrization (deleting one of each $(i,j) / (j,i)$ pair) biases toward whichever FIPS appears first in the source dataframe, so all CBSA-level aggregates are computed using the conserved-total identity $\textbf{OCOC}_{C} = \textbf{Total}_{C} - (\textbf{ICIC}_{C} + \textbf{ICCC}_{C})$ instead.

---

## Methodology

### 1. From SCI to absolute connection counts

Meta publishes the SCI but not the raw connection counts. The two are, however, tied together by the published normalization. For two distinct counties $i \neq j$, the heterogeneous SCI satisfies $\text{SCI}_{i,j} = \text{FB Conn.}_{i,j} \,/\, (|S_i|\cdot |S_j|)$. For the within-county case, the homogeneous SCI satisfies $\text{SCI}_{i,i} = \text{FB Conn.}_{i,i} \,/\, (|S_i|\cdot (|S_i| - 1))$. The within-county denominator uses $|S_i|(|S_i| - 1)$ rather than $\binom{|S_i|}{2}$ because the SCI is published with both $(i,j)$ and $(j,i)$ entries — each Facebook friendship is double-counted in the directed edge list, which exactly cancels the factor of two from the unordered-pair formula. Inverting these two relations gives the per-county and per-county-pair connection counts that feed every downstream step.

### 2. Aggregating to CBSA / MSA / muSA

CBSA-level connections are built up from the county-level connections of each CBSA's constituent counties. The three CBSA-level types — ICIC (within-CBSA, within-county), ICCC (within-CBSA, between-county), and OCOC (outside-CBSA) — partition the total degree of a CBSA, so we can recover OCOC via subtraction rather than enumerate every cross-CBSA edge: $\textbf{OCOC}_{C} = \textbf{Total}_{C} - (\textbf{ICIC}_{C} + \textbf{ICCC}_{C})$. That avoids ever materializing the ~10M-row symmetric edge list grouped by complement covering, which is the step that ran out of memory in the earlier community-identification attempts.

### 3. Coverage rescaling and normalization (per *Schläpfer et al., 2014*)

Because we only see the Facebook-active subset of each county's population — and because that coverage rate $s_i = |S_i|/N_i$ varies geographically — fitting a power law directly on $k$ vs. $N$ would confound the scaling we actually care about with a county-by-county sampling-rate confound. The *Schläpfer* correction divides the cumulative degree by coverage so that the rescaled $K_{r,\,i} = k_{i,\,t} / s_i$ is proportional to the true population-level degree, then normalizes both $K_r$ and $N$ by their respective averages so the cloud of points is centered at $(1, 1)$ and the regression becomes scale-invariant. For MSA-only and muSA-only fits the averages are recomputed *within* each subpopulation, not over all CBSAs, so the normalization is internal to the size band being fit and does not inherit the metropolitan / micropolitan size mismatch.

### 4. Power-law fit

A power law $k = N^{\beta} \epsilon$ becomes linear in log-log space:

$$
\log_{10}\!\Big(\frac{K_r}{\langle K_r \rangle}\Big) = \beta \cdot \log_{10}\!\Big(\frac{N}{\langle N \rangle}\Big) + \gamma + \log_{10}\epsilon
$$

I fit this via OLS in `statsmodels` for each (GEOID type) × (network type) combination, recording $\beta$, $\gamma$, adjusted $R^2$, RMSE, RSE, and 95% confidence intervals on both parameters via the $t$-distribution. The slope $\beta$ is the quantity of interest: superlinear ($\beta > 1$) means bigger places generate disproportionately more connections per person, linear ($\beta = 1$) means the per-capita rate is constant, and sublinear ($\beta < 1$) means bigger places have proportionally fewer connections per person.

---

## Results: Scaling

### The scaling table

The fitted $\beta$ values across all four GEOID levels and all three network types are summarized in Table 1 below. All confidence intervals are 95% via the $t$-distribution with $N - 2$ degrees of freedom.

![Image 1: Results Table](image/README/1769003772068.png)

> *Table 1: Network regression results by GEOID type and network type. Rescaled degree, population, and user estimates shown separately.*

Reading the table, three patterns assert themselves with remarkable consistency across the four GEOID resolutions. The first is that **inner connections are superlinear at every level**, with $\beta_{\text{Inner}}$ landing between 1.054 and 1.082 and the 95% CI sitting strictly above 1 at the County, CBSA, and MSA levels. This is the same qualitative finding *Schläpfer et al.* report for European face-to-face / phone interaction data: the larger the place, the more than proportionally its residents interact with each other. Our point estimates sit at the low end of their range ($\beta \approx 1.1\!-\!1.2$ in their European urban data), which I read as a real but modest superlinearity at U.S. county resolution — plausibly attenuated by the fact that Meta's "within-county" Facebook ties include many that the European face-to-face data would not capture, like reconnected high-school classmates who no longer live nearby but happen to share a county FIPS.

The second pattern is that **outgoing connections are sublinear at every level**, with $\beta_{\text{Outgoing}} \in [0.914, 0.945]$ and CIs again excluding 1 at the County, CBSA, and MSA resolutions. Larger places generate proportionally *fewer* outgoing per-capita ties as they grow, which makes sense if denser local social networks crowd out long-range ties: in a 5,000-person town, almost everyone you know lives somewhere else, but in a 5,000,000-person metro, your network is more likely to be locally saturated.

The third pattern is that **total connections sit just below linearity**, with $\beta_{\text{Total}} \in [0.969, 0.982]$ — the superlinearity of the inner ties almost, but not quite, cancels the sublinearity of the outgoing ties. The total exponent is the closest of the three to exact linearity, which is intuitive: once you sum within- and across-place connections, the total degree of a place should be *roughly* proportional to its population, and indeed it is. The slight residual sublinearity in the total exponent is the part of the urban-scaling literature that I find genuinely interesting in this data — it implies that, for online social ties at U.S. resolution, the within-place and cross-place effects are nearly but not exactly balanced, and the direction of imbalance is that connectivity *underscales* (slightly) with population.

A separate point worth registering is the **stability of the exponents across GEOID resolution**. As we walk County → CBSA → MSA, the three slopes barely move; this is exactly what we'd want from a real property of the social network rather than an artifact of how the geographic partition was drawn. The one exception is the muSA-only fit, where the $R^2$ values drop from the $\approx 0.92\!-\!0.96$ range at the MSA level to roughly $0.69\!-\!0.82$. The muSA Total CI ($[0.931, 1.007]$) actually includes 1.0 — meaning that, for total connectivity in micropolitan areas, we cannot statistically reject pure linearity. This is consistent with micropolitan areas spanning a narrower population range and so giving the regression less leverage; it's not surprising, but it's worth flagging.

### County-level regressions

The county-level fit is the most data-rich (3,141 points) and the cleanest visually. The superlinear Inner slope, sublinear Outgoing slope, and almost-linear Total slope are all visible by eye, and the 95% confidence band is narrow enough that the deviations from $\beta = 1$ are unambiguous.

![Image 2: County-level scaling regressions](plots/regressions/county_connection_regressions.png)

> *Figure 1: Log-log OLS fits of normalized rescaled cumulative degree vs. normalized population at the County level. Left: Inter-County (within-county). Center: Outgoing (cross-county). Right: Total. Inset boxes report $\beta$, $\gamma$, $R^2$, RMSE, RSE, sample size, and the average ESRI user / population estimates.*

### CBSA-level regressions

Aggregating up to CBSAs collapses the 3,141 points into 917 and improves visual separability — but the slopes barely move, which is the point. The qualitative story (inner superlinear, outgoing sublinear, total just-below-linear) is preserved intact.

![Image 3: CBSA-level scaling regressions](plots/regressions/cbsa_connection_regressions.png)

> *Figure 2: Log-log fits at the CBSA level (combined Metropolitan + Micropolitan).*

### MSA-only regressions

Restricting to MSAs (the 381 metropolitan-only CBSAs) further reduces sample size but tightens the signal-to-noise for the metro-only band of populations. This is the regime where I'd expect the urban-scaling effect to be cleanest, and indeed the Inner slope creeps up to $\beta = 1.082$ and the Outgoing slope down to $\beta = 0.914$ — the most extreme deviations from linearity anywhere in the study.

![Image 4: MSA-level scaling regressions](plots/regressions/msa_connection_regressions.png)

> *Figure 3: MSA-only fits.*

---

## Results: Geographic Structure (Choropleths)

Before collapsing everything to a single regression slope, it's worth confirming that the connection counts have the geographic structure we'd expect. The raw-count maps light up exactly where they should — the BosWash corridor, the California megaregion, the Texas Triangle, the Front Range, Florida — and dim out across the High Plains and rural Mountain West. The Outgoing map is denser than the Inner map (because every county has more cross-county ties than within-county ties), but both show the same metropolitan weighting.

![Image 5: County-level connection choropleths](plots/choropleths/county_connections_choropleth.png)

> *Figure 4: County-level choropleths of Inter-County (within), Outgoing (cross-county), and Total connections. Color is on a log scale; ticks are formatted in raw connection counts for readability.*

The companion user / population / coverage map is the more diagnostically interesting of the two. The user and population maps share the metro / coastal weighting of Figure 4, but the **coverage map** — the right-most panel — is the one that justifies the rescaling step. Coverage spans roughly 0.34 to 0.63 across counties, with the highest values concentrated in dense, younger, more-online counties (urban Northeast, Pacific coast, parts of the Mountain West) and the lowest values in much of the rural South and Plains. That is exactly the kind of geographically heterogeneous undercount that the *Schläpfer et al.* rescaling step was designed to absorb: if coverage were spatially uniform, dividing by it would just shift the intercept; because it's not, the rescaling actually changes the *slope* in ways that matter.

![Image 6: County-level user, population, and coverage estimates](plots/choropleths/county_popstats_choropleth.png)

> *Figure 5: County-level ESRI user-count estimates $|S_i|$ (left, log scale), population estimates $N_i$ (center, log scale), and coverage estimates $s_i = |S_i|/N_i$ (right, linear, range ~0.34-0.63).*

The qualitative confirmation that the rescaling has done its job comes from the per-user choropleth below. Where the raw-count map of Figure 4 had a color range spanning roughly six decades and was visually dominated by metro counties, the per-user map sits on a single decade and the remaining geographic structure is much more subtle. That subtle residual structure is the *real* signal — it's what's left after we strip out the population-driven baseline.

![Image 7: Per-user connection choropleths](plots/choropleths/county_peruser_choropleth.png)

> *Figure 6: County-level connections **per Facebook user**. Note the much narrower color range relative to Figure 4 — the regional heterogeneity that remains is the true per-capita signal, not the population baseline.*

---

## Results: Distributions

For each variable, I fit a Kernel Density Estimate (KDE, Gaussian kernel) as the non-parametric reference PDF and compare four parametric candidates — Normal, Log-Normal, Skew-Normal, and a Generalized Pareto fit to the top 75% tail — by RMSE against the KDE at the bin centers.

![Image 8: County-level SCI histograms](plots/histograms/connectivity/sci/sci_histograms.png)

> *Figure 7: $\log_{10}$ SCI distributions per county for Inner (left), Outgoing (center), and Total (right) connections, with KDE (black) and parametric overlays.*

The most important thing the distributions tell us is that the **log-transformed** Inner, Outgoing, and Total connection counts are close to Normal in shape — i.e., the raw counts are roughly **log-normal**, which is the canonical distribution arising from multiplicative growth processes. That is precisely the assumption the power-law fit depends on: if $k = N^{\beta} \cdot \epsilon$ with multiplicative noise $\epsilon$, then $\log k$ is additive in $\log N$, and the OLS fit is well-posed. The fitted LogNormal PDF has the lowest RMSE vs. KDE for the bulk of variables, with the SkewNormal occasionally edging it out for the more right-skewed populations. The Generalized Pareto fit to the top tail captures the heavy upper end (the few outsize metros that dominate the absolute connection counts) without forcing the bulk distribution into a heavy-tailed family. The full set of distribution plots — broken out by County, CBSA, MSA, and muSA, for both raw counts and per-capita / per-user ratios — lives under [`plots/histograms/`](plots/histograms/).

---

## Results: Demographic Profiles

The demographic section zooms from the national scaling picture down to a handful of NYC census blocks — specifically the Hamilton Beach + Howard Beach blocks in Queens (FIPS prefix `360810884006`), and Red Hook in Brooklyn as a comparison. Both communities sit in FEMA-designated coastal flood zones; both are points of contact for CLIMA's homeowner interviews.

### Hamilton & Howard Beach (Queens)

The Hamilton + Howard Beach age/sex pyramid is the first place the focus community's distinctiveness shows up. The pyramid skews noticeably older than the NYC baseline: the bulk of residents are in the 30-64 range, with under-representation in the 18-29 cohort relative to the citywide average. This is the canonical demographic shape of a "settled" outer-borough community where residents have lived for a long time and turnover is low.

![Image 9: HB pop pyramid](plots/demographics/bar%20charts/allBlocks/hb_allBlocks_popPyramid.png)

> *Figure 8: Age/sex population pyramid for the selected Hamilton + Howard Beach census blocks (bars), with NYC census-block averages overlaid as dots.*

The race/ethnicity composition tells a complementary story: the selected blocks are markedly more White non-Hispanic (~47% vs. ~53% NYC), markedly more Hispanic / Latino (~36% vs. ~20% NYC), and markedly less Black non-Hispanic (~6% vs. ~14% NYC). The high Hispanic share is the demographic feature most likely to surprise readers who think of Queens waterfront communities through the older "Italian and Irish working-class" lens; that demographic has shifted substantially over the last two decades.

![Image 10: HB race distribution](plots/demographics/bar%20charts/allBlocks/hb_allBlocks_raceDistribution.png)

> *Figure 9: Race / ethnicity distribution for Hamilton + Howard Beach (bars) vs. NYC average (dots).*

Household size and occupancy round out the picture. Hamilton + Howard Beach is shifted toward 3- and 4-person households relative to the citywide mix and away from 1-person households (~23% vs. ~30% NYC) — consistent with a family-oriented owner-occupied profile. Overall occupancy tracks the NYC baseline very closely (~90% occupied either way).

![Image 11: HB household size](plots/demographics/bar%20charts/allBlocks/hb_allBlocks_householdsizeDistribution.png)

> *Figure 10: Household size distribution.*

![Image 12: HB occupancy](plots/demographics/bar%20charts/allBlocks/hb_allBlocks_occupancyDistribution.png)

> *Figure 11: Occupied vs. vacant housing.*

The single most consequential variable for CLIMA's homeowner-focused mobility modeling is in Figure 12. Hamilton + Howard Beach is roughly 63% owner-occupied, against an NYC-wide block average of about 51%. That 12-percentage-point gap is the demographic feature that determines whether the population class CLIMA cares about — homeowners under coastal flood risk — is even *present* in a given community at meaningful density.

![Image 13: HB tenure](plots/demographics/bar%20charts/allBlocks/hb_allBlocks_housingtenureDistribution.png)

> *Figure 12: Owner- vs. renter-occupied housing.*

Finally, the educational-attainment and household-income pie charts give a socioeconomic snapshot. Education is bottom-heavy: roughly 30% of adults have no HS diploma, ~26% have only a HS diploma, and only ~24% hold a bachelor's degree or higher. Income is concentrated in the $60k-$85k bracket (~48%), with a long thin upper tail.

![Image 14: Hamilton Beach education + income](plots/demographics/pi%20charts/hamBeach_pi.png)

> *Figure 13: Hamilton Beach educational attainment (left) and household income (right).*

### Red Hook (Brooklyn) — for comparison

Red Hook uses the same education / income bins as Hamilton Beach so the two are directly comparable. The differences are stark. Red Hook has substantially more bachelor's-and-above (~32% vs. ~24% in HB), a notably bimodal income distribution with a large mass at <\$10k *and* a fat upper tail (~25% of households earn over \$100k), and a noticeably more even spread across the middle income brackets. The two communities sit in similar FEMA flood-zone designations, but the social exposure to a flood event is quite different in each — Hamilton Beach is a relatively homogenous middle-income owner-occupied community; Red Hook is a much more bimodal community that includes both deeply low-income public-housing renters *and* a layer of higher-income recent owner-occupants.

![Image 15: Red Hook education + income](plots/demographics/pi%20charts/redHook_pi.png)

> *Figure 14: Red Hook educational attainment (left) and household income (right).*

### Hamilton Beach vs. Red Hook, at a glance

| Variable                       | Hamilton + Howard Beach | Red Hook (block groups) |
| ------------------------------ | ------------------------ | ----------------------- |
| Bachelor's degree or higher    | ~24%                     | ~32%                    |
| No HS diploma                  | ~30%                     | ~18%                    |
| Median income bracket          | \$60k-\$85k (~48%)       | \$10k-\$25k (~17%)      |
| Households earning \$100k+     | ~12%                     | ~25%                    |
| Housing tenure (owner-occupied, NYC comparison) | High (~63%, vs. ~51% NYC) | Lower (predominantly renter) |
| Dominant household size mode   | 2- and 3-person          | 1- and 2-person         |

The takeaway is that "coastal homeowner under flood risk" is not a uniform agent class. It spans communities with very different baselines for income, education, and tenure, and any compartmental or radiation-style mobility model built on top of the scaling exponents in this notebook eventually has to be calibrated to those differences.

---

## Grounding the Model: Demographic Summaries & Resident Interviews

The quantitative pipeline above gives a population-level picture of who lives in Hamilton Beach, Howard Beach, and Red Hook. The CLIMA modeling work, though, is ultimately about *what those residents do* when faced with repeated and worsening coastal flood events — and that question is one the demographic tables cannot answer on their own. To get at it, the project also includes semi-structured interviews with homeowners in Hamilton Beach, Howard Beach, and Red Hook, conducted under the CLIMA umbrella and with the relevant CITI training completed and transcriptions stored separately from this repo's quantitative artifacts.

The interviews are intentionally open-ended around a small set of themes: prior flood-event experience (Hurricane Sandy in particular, since it was the formative event for all three communities), perception of future flood risk and the time horizon over which residents expect to make stay-or-leave decisions, the role of neighbors and longstanding community ties in those decisions, trust in (or skepticism of) city- and federal-level mitigation programs, and the practical financial constraints — mortgage status, equity, insurance availability, and the asymmetry between selling into a flood-zone market and buying out of one — that frame what "relocate" even means in practice. Across both communities, recurring observations include a strong sense of place attachment that is *not* well captured by income or tenure variables alone, a generational layer to how flood risk is interpreted (older residents who have weathered multiple events frame the risk differently than newer homeowners), and a clearly information-network-mediated decision process in which neighbors' visible decisions to repair, raise, or sell are themselves an input to one's own thinking.

Those observations directly shape several modeling choices that the quantitative side of the notebook would otherwise leave under-justified. The choice to use a *compartmental* model — partitioning the population into discrete behavioral classes rather than tracking individuals — is anchored by the interview-level evidence that residents largely talk about their decisions in categorical terms (stay-and-repair, stay-and-elevate, sell-and-relocate-nearby, sell-and-relocate-far) rather than as continuous risk-utility tradeoffs. The choice to invoke the *mean-field hypothesis* (treating individuals within a compartment as interchangeable) is defensible at the population level precisely because the interviews suggest the relevant dispersion is *between* compartments, not within them; two homeowners with the same demographic profile and the same compartment label appear, qualitatively, to behave similarly. The choice to read the scaling exponents as parameters that govern how fast *information about flood events* propagates through the network — rather than as direct predictors of mobility — is also interview-grounded: residents repeatedly described becoming aware of their neighbors' decisions through social ties, and only after that becoming serious about their own. And the choice to extend the *Simini et al.* radiation framework with an "available comparable homes" opportunity surface, rather than the original "available jobs" surface, is grounded in interviewees consistently framing relocation as constrained by housing availability and affordability, not by employment-driven opportunity in the labor-market sense.

The demographic figures and the interviews are best read together as the two complementary halves of the same input layer: the demographics tell us *who is there*, and the interviews tell us *how they think about leaving*. The interviews themselves remain in restricted CLIMA storage rather than this public repo, but they are the qualitative scaffold that makes the modeling work above more than just a curve-fitting exercise.

---

## Interpretation: Why This Matters

Putting the scaling and demographic pieces side by side, the national scaling result implies that per-capita social connectivity is *not* constant across population scale. Larger cities and CBSAs have superlinear within-region connectivity, sublinear cross-region connectivity, and near-linear total connectivity. In modeling terms, this means the effective transmission rate of any process spreading through the social graph — flood-risk information, neighbor-mover signals, policy uptake — depends on the population scale of the place you're standing in, not just on per-capita averages. A coastal block inside a large CBSA (such as Hamilton Beach inside New York-Newark-Jersey City) is embedded in a denser local network than the same-sized population would be in a micropolitan area, and the effective *speed* of decision-cascades is therefore regionally heterogeneous in a quantifiable way.

The demographic snapshots show what those nodes actually look like up close: predominantly owner-occupied, older-than-citywide, with a HS-or-less education plurality in Hamilton Beach; more bimodal in income, more educated, and more renter-dominant in Red Hook. The interview material then describes how those very different starting points map onto very different decision processes around stay-or-leave behavior. Any model that treated "coastal homeowner" as a uniform agent class would erase exactly the heterogeneity that the demographic plots and interview themes make visible.

The combination of the three layers — macroscopic scaling exponents, block-level demographic context, and qualitative interview material — is the empirical scaffolding for the next phase of the CLIMA modeling work. That phase couples these inputs to FEMA's National Risk Index inside a *Simini*-style mobility framework with "available comparable housing" as the opportunity surface, parameterized by the connectivity scaling above and calibrated to the compartmental behaviors that the interviews surface.

---

## Caveats

A few practical limitations sit behind the headline results. Most importantly, county-level Facebook MAU is not publicly reported by Meta. ESRI is our best stand-in, but comparing our derived average $s_i$ against Meta's published Q4 '22 MAU figure suggests the ESRI-derived coverage runs ~17% below the true MAU coverage. A uniform underestimate would shift the intercept $\gamma$ but leave $\beta$ unchanged in expectation; however, because the ESRI MAU model is itself geographically heterogeneous (undercounting more in certain county types than others), there is a second-order effect on $\beta$ that is harder to bound. The most plausible direction is a small decrease in $\beta$ — i.e., the true scaling is slightly *more* sublinear than what we report for Outgoing and Total.

The SCI itself is noised: Gaussian noise in $\pm[0, 1]$ is added by Meta to each connection count for privacy. This is invisible for large counties but can dominate the signal for counties just above the 50,000-user reporting threshold, which is part of why the muSA confidence intervals are wider than their MSA counterparts. The ESRI MAU column is itself a model output, not a direct count — so two layers of estimation (Meta's true MAU → ESRI's modeled MAU → our $|S_i|$) sit between the underlying truth and the regressor.

On the demographic side, the figures are explicitly descriptive. Two communities is two communities; Hamilton Beach + Howard Beach and Red Hook are case studies that ground the modeling work in concrete examples, not a representative sample of coastal NYC. Likewise, the interview material is *qualitative* and is used in this README to motivate modeling choices, not as a quantitative input. Anyone wanting to make stronger claims about NYC waterfront communities as a whole would need to extend both layers — more blocks, more interviews — to the full set of FEMA 100-year and 500-year designated tracts.

Finally, the 2021 SCI is paired with 2022 ESRI estimates. The temporal mismatch is small and Meta has shown that the SCI is stable across longer time horizons, but it is a mismatch.

---

## Project Layout

```
clima/
|-- clima.ipynb                  # Annotated end-to-end analysis notebook (full details live here)
|-- README.md                    # This file
|-- clima_poster.pdf             # Conference poster summarizing the work
|-- clima_figures.pdf            # Full-resolution PDF of every figure produced
|-- scripts/
|   |-- data/data_cleaning.py            # SCI + ESRI + crosswalk -> export CSVs
|   `-- visualization/
|       |-- choropleths/choropleth.py
|       |-- histograms/histograms.py
|       |-- regression and ests/regression.py
|       `-- demographics/{demographics,py_chart}.py
|-- source/                      # Raw input data (SCI, users, crosswalk, Census, shapefiles)
|-- export/                      # Cleaned CSVs consumed by visualization scripts
|-- plots/                       # PNG outputs (choropleths, histograms, regressions, demographics)
`-- image/                       # README and notebook images (poster, coverage screenshots, etc.)
```

The notebook is the single source of truth — every plot in this README was rendered by code that also appears in [`clima.ipynb`](clima.ipynb). The standalone Python scripts under [`scripts/`](scripts/) are extracted directly from the notebook cells if you'd rather run them outside Jupyter. For a one-page visual summary, see [`clima_poster.pdf`](clima_poster.pdf); for the full PDF of every figure at full resolution, see [`clima_figures.pdf`](clima_figures.pdf).

---

## References

### Meta SCI Resources

- [SCI Homepage](https://data.humdata.org/dataset/social-connectedness-index)
- [SCI Methodology](https://dataforgood.facebook.com/dfg/docs/methodology-social-connectedness-index)
- [SCI Docs](https://data.humdata.org/dataset/e9988552-74e4-4ff4-943f-c782ac8bca87/resource/a0c37eb4-b45c-436d-b2b2-c0c9b1974318/download/documentation-fb-social-connectedness-index-october-2021.pdf)
- [County-to-County SCI Dataset](https://data.humdata.org/dataset/e9988552-74e4-4ff4-943f-c782ac8bca87/resource/c59fd5ac-0458-4e83-b6be-5334f0ea9a69/download/us-counties-us-counties-fb-social-connectedness-index-october-2021.zip)

### Meta Official Regional Coverage Estimates

- [Meta Q4 '22 Earnings Presentation](https://s21.q4cdn.com/399680738/files/doc_financials/2023/q4/Earnings-Presentation-Q4-2023.pdf)

### ESRI Facebook User Estimates

- [ESRI Data](https://nyuds.maps.arcgis.com/home/item.html?id=14a2fb32e22b4fe5ab9d884c9e994075)
- [ESRI Documentation](https://demographics5.arcgis.com/arcgis/rest/services/USA_MPI_1_2022/MapServer/7)

### Crosswalk

- [County-MSA-CSA Crosswalk](https://www.bls.gov/cew/classifications/areas/county-msa-csa-crosswalk.html)

### Census / Demographic Data

- [IPUMS NHGIS](https://www.nhgis.org/) — 2020 DHC and ACS extracts
- [TIGER/Line Shapefiles (2021)](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

---

### Bibliography

1. Bailey, Michael, Rachel Cao, Theresa Kuchler, Johannes Stroebel, and Arlene Wong. **"Social Connectedness: Measurement, Determinants, and Effects."** *Journal of Economic Perspectives* 32, no. 3 (August 2018): 259-280. DOI: [10.1257/jep.32.3.259](https://doi.org/10.1257/jep.32.3.259)
2. Schläpfer, M., Bettencourt, L. M. A., Grauwin, S., Raschke, M., Claxton, R., Smoreda, Z., West, G. B., & Ratti, C. (2014). **The scaling of human interactions with city size.** *Journal of the Royal Society Interface*, **11**(98), 20130789. DOI: [10.1098/rsif.2013.0789](https://doi.org/10.1098/rsif.2013.0789)
3. Simini, F., González, M., Maritan, A., et al. **A universal model for mobility and migration patterns.** *Nature* **484**, 96-100 (2012). DOI: [10.1038/nature10856](https://doi.org/10.1038/nature10856)

---

al fin :]
