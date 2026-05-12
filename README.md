# Random Forest Classification of TCGA-STAD Molecular Subtypes
Stomach adenocarcinoma (STAD) is a biologically heterogeneous cancer with multiple molecular subtypes, including EBV-positive, MSI, GS, and CIN, each characterized by distinct genomic patterns. This project investigates the classification of STAD subtypes using Random Forest models trained on TCGA gene expression data, comparing both multiclass and binary classification approaches. Previous experiments showed that binary MSS vs. MSI classification was suboptimal due to the molecular similarity between MSS-L and MSI tumors, highlighting the challenges of subtype discrimination in high-dimensional transcriptomic datasets. This repository serves as a dedicated development version created for the course *Developing Software as a Product*, taught by Dr. Julija Pečerska.

## State at the Start of the Project
The repository started in an unstructured state: a single main branch, no README, poorly described commits, and all code contained in one Jupyter Notebook. The main issues were a complete lack of modularisation, excessive and inconsistent commenting (including mixed English/German), and unclear naming conventions that became increasingly cryptic over time.

## Refactoring Priorities
Critical work included restructuring the repository, adding this README, converting the Notebook to modular Python scripts, cleaning up comments and replacing them with docstrings, standardizing naming, and introducing basic tests. Full test coverage and CI/CD were considered important but out of scope. Interactive visualizations and extended benchmarking were noted as optional stretch goals.

## Repository Structure
The repository layout is as follows:
```text
data/              TCGA-STAD datasets and subtype annotations
snapshot/          archived snapshots of the original project state
tests/             pytest-based testing scripts
Clean_Code.py      cleaned and refactored analysis pipeline
README.md          project overview and usage instructions
CONTRIBUTING.md    contribution and workflow guidelines
.gitignore         ignored system and environment-specific files
```

## CI/CD Pipeline
The repository includes a basic CI/CD workflow implemented with GitHub Actions.

Continuous Integration (CI) is used to automatically validate code quality through:
- pytest test execution
- ruff linting
- black formatting checks
- coverage analysis with pytest-cov

Continuous Delivery (CD) extends the workflow by:
- automatically training Random Forest models
- storing trained models as GitHub Artifacts
- publishing the package to TestPyPI
- building executables for Linux, macOS, and Windows

The workflows are executed automatically on pushes and pull requests to feature, develop, and main branches.

## Running the Analysis
Clone the repository:
```bash
git clone https://github.com/luludoaido/Dev_SAP.git
cd Dev_SAP
```

The project was developed using Python and standard machine learning libraries, including pandas, numpy, scikit-learn, matplotlib, seaborn, and pytest.

Run the main analysis script:
```bash
python Clean_Code.py
```

## Data
The repository contains TCGA-STAD gene expression data together with molecular subtype and clinical annotation files used for Random Forest classification analysis.

Included dataset files:
- TCGA-STAD_gene_expression_cpm.csv
- TCGA-STAD clinical data.csv
- TCGA-STAD subtype data.csv
- TCGA-STAD_subtypes.csv

## Authors
- Luana da Silva do Aido
- Serna Sürer
- Luca Baldi

Course: Developing Software as a Product
ZHAW

## License
This project is licensed under the MIT License – see the LICENSE file for details.
