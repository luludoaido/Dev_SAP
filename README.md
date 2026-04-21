# Trackmodule_1_RF_TCGA-STAD

## Introduction
Stomach adenocarcinoma (STAD) is a biologically heterogeneous tumour whose major molecular subtypes (EBV-positive, MSI, GS, and CIN) each exhibit distinct genomic characteristics, making accurate classification both challenging and clinically relevant (Goldenring & Nam, 2010). This project builds on a previous project in which Random Forest classifiers were used to classify STAD subtypes from TCGA gene expression data (https://github.com/luludoaido/Trackmodule_1_RF_TCGA-STAD). A multiclass and a binary (MSS vs. MSI) approach were compared, with the binary approach proving suboptimal due to the molecular similarity between MSS-L and MSI tumors (Han et al., 2022). This repository is a dedicated development version created for the course "Developing Software as a Product" teached by Dr. Julija Pečerska.

## State at the Start of the Project
The repository started in an unstructured state: a single main branch, no README, poorly described commits, and all code contained in one Jupyter Notebook. The main issues were a complete lack of modularisation, excessive and inconsistent commenting (including mixed English/German), and unclear naming conventions that became increasingly cryptic over time.

## Refactoring Priorities
Critical work included restructuring the repository, adding this README, converting the Notebook to modular Python scripts, cleaning up comments and replacing them with docstrings, standardizing naming, and introducing basic tests. Full test coverage and CI/CD were considered important but out of scope. Interactive visualizations and extended benchmarking were noted as optional stretch goals.

## References
Goldenring, J. R., & Nam, K. T. (2010). Oxyntic Atrophy, Metaplasia, and Gastric Cancer. In Progress in Molecular Biology and Translational Science (Vol. 96, pp. 117–131). Academic Press. https://doi.org/10.1016/B978-0-12-381280-3.00005-1

Han, S., Chok, A. Y., Peh, D. Y. Y., Ho, J. Z.-M., Tan, E. K. W., Koo, S.-L., Tan, I. B.-H., & Ong, J. C.-A. (2022). The distinct clinical trajectory, metastatic sites, and immunobiology of microsatellite-instability-high cancers. Frontiers in Genetics, 13. https://doi.org/10.3389/fgene.2022.933475

Zhang, W. (2014). TCGA divides gastric cancer into four molecular subtypes: Implications for individualized therapeutics. Chinese Journal of Cancer, 33(10), 469–470. https://doi.org/10.5732/cjc.014.10117

