# Random Forest Classification of TCGA-STAD Molecular Subtypes

Stomach adenocarcinoma (STAD) is a biologically heterogeneous cancer with multiple molecular subtypes, including EBV-positive, MSI, GS, and CIN, each characterized by distinct genomic patterns. This project investigates the classification of STAD subtypes using Random Forest models trained on TCGA gene expression data, comparing both multiclass and binary classification approaches. Previous experiments showed that binary MSS vs. MSI classification was suboptimal due to the molecular similarity between MSS-L and MSI tumors, highlighting the challenges of subtype discrimination in high-dimensional transcriptomic datasets. This repository serves as a dedicated development version created for the course *Developing Software as a Product*, taught by Dr. Julija Pečerska.

## State at the Start of the Project
The repository started in an unstructured state: a single main branch, no README, poorly described commits, and all code contained in one Jupyter Notebook. The main issues were a complete lack of modularisation, excessive and inconsistent commenting (including mixed English/German), and unclear naming conventions that became increasingly cryptic over time.

## Refactoring Priorities
Critical work included restructuring the repository, adding this README, converting the Notebook to modular Python scripts, cleaning up comments and replacing them with docstrings, standardizing naming, and introducing basic tests. Full test coverage and CI/CD were considered important but out of scope. Interactive visualizations and extended benchmarking were noted as optional stretch goals.
