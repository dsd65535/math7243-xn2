MATH7243 XN2
============

This repo contains code related to the MATH7243 XN2 project.

Installation
------------

The dependencies for this project are defined in `pyproject.toml`.
However, because of the way that TensorFlow marks the optional CUDA dependency, `pip` may fail.

Installation has been tested with poetry (https://python-poetry.org/).
Install poetry and then simply run `poetry install` in this directory.

Datasets
--------

The data used for this project requires an EULA.
Please download `sample_info.csv`, `CCLE_expression.csv` and `CRISPR_gene_effect.csv`
from the `DepMap Public 22Q2` dataset at https://depmap.org/portal/data_page/?tab=allData.
Place all three files in `cache/22Q2`:
```
cache/
└── 22Q2
    ├── CCLE_expression.csv
    ├── CRISPR_gene_effect.csv
    └── sample_info.csv
```

Usage
-----

Once the package is install and the datasets are downloaded,
run `python math7243_xn2`.
This will generate all results in the report.
