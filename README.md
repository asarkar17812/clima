# Scaling of Social Connections & Demographic Analysis

## Ayush Sarkar: 5/15/2025 - 7/11/2025 | CLIMA w/ Dynamical Systems Lab @ NYU

---

## Objectives:

---

CiviL Infrastructure research for climate change Mitigation and Adaptation (CLIMA) is a research effort focused on infrastructure research to help develop equitable and feasible solutions to the increasingly urgent threats posed by climate change, specifically through the mitigation of damages and adaptation to hazards and changes across coastal communities worldwide.

This leg of the CLIMA project is an interdisciplinary research project that aims to model the effects of flood risk on coastal communities through a detailed investigation of the social networks, as well as other factors, in an attempt to create a model that is more capable of capturing human mobility phenomena, especially amongst homeowners. By using a modified compartmental model, we can partition the population into sets and utilize the mean-field hypothesis to treat individuals as identical, thus allowing us to focus on the population's dynamics instead of individual-level cognition. That being said, to better appreciate the unique perspectives of individuals and communities that most directly face these threats, this project also seeks to include qualitative information about the network extracted through interviews with homeowners of coastal communities here, in NYC. Furthermore, demographic breakdowns of the sample coastal communities enable us to increase our understanding of the context and everyday lives of the homeowners within these communities and issues of inequity.

This repo/notebook specifically focuses on the scaling of social connections with population in counties and CBSAs, and demographic analyses. Specifically, it contains the figures:

* Linear Regression of $log_{10}$(Social Connections) vs $log_{10}$(Population)
  * Rescaled & Normalized (*Schlapfer et. al*)
  * County & CBSA plots by network type
* Choropleths:
  * Inter-County, Outer-County, and Total County Connections per User and per Capita
  * County User, Population, and Coverage Estimates
* Histograms with non-parametric and parametric fitted PDFs:
  * County:

    * Inter-County, Outer-County, and Total County Connections
    * Inter-County, Outer-County, and Total County Connections per User and per Capita
    * User, Population, and Coverage Estimates
  * CBSA:

    * Inter-Covering, Outer-Coveirng, and Total CBSA Connections
    * Inter-Covering, Outer-Coveirng, and Total CBSA Connections per User and per Capita
    * Aggregated User, Population, and Coverage Estimates for CBSAs
* Age/Sex Population Pyramids w/ NYC Averages (Selected Hamilton Beach Census Blocks)
* Race/Ethnicity Distribution w/ NYC Averages (Selected Hamilton Beach Census Blocks)
* Persons per Household (1-5+) Distribution w/ NYC Averages (Selected Hamilton Beach Census Blocks)
* Housing Occupation Status Distribution w/ NYC Averages (Selected Hamilton Beach Census Blocks)
* Housing Tenure Distribution w/ NYC Averages (Selected Hamilton Beach Census Blocks)

---

![asarkar_nyu_ugsrp_poster](https://github.com/user-attachments/assets/de930506-2524-41f4-acb0-c01ba97a09e1)![1769003772068](image/README/1769003772068.png)

---

## References:

---

### Meta SCI Resources

- [SCI Homepage](https://data.humdata.org/dataset/social-connectedness-index)
- [SCI Methodology](https://dataforgood.facebook.com/dfg/docs/methodology-social-connectedness-index)
- [SCI Docs](https://data.humdata.org/dataset/e9988552-74e4-4ff4-943f-c782ac8bca87/resource/a0c37eb4-b45c-436d-b2b2-c0c9b1974318/download/documentation-fb-social-connectedness-index-october-2021.pdf)
- [County-to-County SCI Dataset](https://data.humdata.org/dataset/e9988552-74e4-4ff4-943f-c782ac8bca87/resource/c59fd5ac-0458-4e83-b6be-5334f0ea9a69/download/us-counties-us-counties-fb-social-connectedness-index-october-2021.zip)

### Meta Official Regional Coverage Estimates

- [Meta Q4 ’22 Earnings Presentation](https://s21.q4cdn.com/399680738/files/doc_financials/2023/q4/Earnings-Presentation-Q4-2023.pdf)

### ESRI Facebook User Estimates

- [ESRI Data](https://nyuds.maps.arcgis.com/home/item.html?id=14a2fb32e22b4fe5ab9d884c9e994075)
- [ESRI Documentation](https://demographics5.arcgis.com/arcgis/rest/services/USA_MPI_1_2022/MapServer/7)

### Crosswalk

- [County–MSA–CSA Crosswalk](https://www.bls.gov/cew/classifications/areas/county-msa-csa-crosswalk.html)

---

### Bibliography

---

1. Bailey, Michael, Rachel Cao, Theresa Kuchler, Johannes Stroebel, and Arlene Wong.**“Social Connectedness: Measurement, Determinants, and Effects.”***Journal of Economic Perspectives* 32, no. 3 (August 2018): 259–280.DOI: [10.1257/jep.32.3.259](https://doi.org/10.1257/jep.32.3.259)
2. Schläpfer, M., Bettencourt, L. M. A., Grauwin, S., Raschke, M., Claxton, R., Smoreda, Z., West, G. B., & Ratti, C. (2014).
   **The scaling of human interactions with city size.**
   *Journal of the Royal Society Interface*, **11**(98), 20130789.
   DOI: [10.1098/rsif.2013.0789](https://doi.org/10.1098/rsif.2013.0789)
3. Simini, F., González, M., Maritan, A., et al.**A universal model for mobility and migration patterns.
   **Nature** **484**, 96–100 (2012).
   DOI: [10.1038/nature10856](https://doi.org/10.1038/nature10856)

---

al fin :]
