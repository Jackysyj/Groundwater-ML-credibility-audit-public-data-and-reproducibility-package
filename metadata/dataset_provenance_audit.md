# External dataset provenance audit

Access date: 11 July 2026. The structured source of record is `dataset_provenance_registry.json`.

## Decision rule

A dataset is verified only when its landing page, version DOI, cite-as text, license terms and local fingerprint are all resolved. Fingerprint acceptance requires at least one hard match and two supporting matches. Similar titles or study regions are not sufficient.

## FrenchPiezo

- **Status:** verified.
- **Landing page:** https://zenodo.org/records/7193812
- **DOI:** version 10.5281/zenodo.7193812; concept 10.5281/zenodo.7193811; version not stated.
- **Cite as:** MBOUOPDA, M. F., GUYET, T., LABROCHE, N., & HENRIOT, A. (2022). *FrenchPiezo: the French mainland groundwater level multivariate time series* [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.7193812
- **License:** CC BY 4.0. Credit, license link and change indication are required.
- **Fingerprint:** official metadata names all three retained CSV files and the `bss`, `p`, `tp`, and `e` fields. The independently downloaded `dataset_stations.csv` is byte-identical to the retained file (772,789 bytes; SHA-256 `a4246c87bc286c0e324ac49fe693afad679db393d1dfd778d99929a4d40887e3`). The repository prose retains an older 1,026-series/January-2021 summary, whereas the deposited companion file used here contains 1,329 eligible wells through 24 July 2021; this discrepancy is recorded rather than silently harmonised.

## Swiss Groundwater Database

- **Status:** verified.
- **Landing page:** https://zenodo.org/records/14260400
- **DOI:** version 10.5281/zenodo.14260400; concept 10.5281/zenodo.14260399; version 1.0.
- **Associated paper:** Collenteur et al., *Scientific Data* (2026), doi:10.1038/s41597-026-07353-6.
- **Cite as:** Collenteur, R., & Moeck, C. (2026). *The Swiss Groundwater Database* (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.14260400
- **License:** CC BY 4.0. The repository also advises commercial users to request original authority data rather than treating the compilation as a substitute.
- **Fingerprint:** the article reports 985 wells, 1980-2023 imputed series, and 1971-August 2023 meteorological coverage. The repository description gives the exact `0_data/{final,raw,processed,gis,imputation}` hierarchy and canton-level imputation files found locally. Local ZIP SHA-256: `4e1d0d5d1625f597020c529fec5413e1aedb0fd07df29ce6c7567e9ca049d91e`.

## Tuscany groundwater drought

- **Status:** verified.
- **Landing page:** https://zenodo.org/records/17491496
- **DOI:** version 10.5281/zenodo.17491496; concept 10.5281/zenodo.17491495; version not stated.
- **Cite as:** Elsaidy, A. (2025). *Groundwater Drought and Water Balance Indicators in Tuscany (2005-2023)* [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.17491496
- **License:** CC BY 4.0. The description identifies SIR Toscana and ISPRA BIGBANG80 as underlying sources.
- **Fingerprint:** the independently downloaded and retained `Tuscany.rar` files are byte-identical (2,409,626 bytes; SHA-256 `d7792fdd774be8e5002d1652f911290ace988374fe6b07b2709de4518db58771`). Both contain the same 114 monthly `TOS*.csv` files with the exact `pr`, `tp`, `td`, `ae`, `rf`, and `ws` fields.
- **Related but not substituted:** doi:10.5281/zenodo.19398188 is the subsequent study record *When drought goes underground: a machine learning approach to groundwater drought*. It is not used as the primary data DOI.

## Search and rejection log

- Zenodo API and direct downloads were inaccessible from the execution environment. Metadata and cite-as records were recovered through DataCite; the French and Tuscany attachments were downloaded by the researcher and checked locally.
- French exact-filename, DataCite, GitHub, HAL, Kaggle and repository searches converged on FrenchPiezo. No similarly named secondary source was substituted.
- The 2026 Scientific Data Swiss descriptor and Zenodo/DataCite record independently agree on the data DOI and core fingerprint.
- `Groundwater Drought and Water Balance Indicators in Tuscany (2005-2023)` is the exact archive source. The later `When drought goes underground` record and other generic Tuscany drought results were rejected as primary data citations.
