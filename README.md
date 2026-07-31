# Reproducible Data Audit for Predictive Maintenance

This repository contains the public computational companion to the manuscript
on data quality, traceability, label uncertainty, and predictive-maintenance
validation in an anonymized manufacturing plant.

`data/synthetic/` contains an entirely synthetic event log that can be used to
execute and inspect the public workflow. The confidential industrial records,
results derived from those records, asset identifiers, work orders, legacy
spreadsheets, dashboards, figures, and row-level reconciliation files are not
included.

## Scientific scope

The workflow:

- audits event-log structure and chronological consistency;
- reconstructs failure-label-to-failure-label intervals;
- evaluates regression models with asset-grouped cross-validation;
- performs an asset-disjoint temporal holdout;
- quantifies uncertainty with an asset-grouped bootstrap;
- stress-tests the analysis under hypothetical bidirectional label
  perturbations; and
- supports aggregate reconciliation with a historical matrix when an
  authorized private copy is supplied locally.

The analysis does **not** establish prospective operational performance or
confirm that every supplied failure label represents a physical failure.

## Repository structure

```text
.
├── data/
│   ├── DICTIONARY.md
│   └── synthetic/
│       ├── README.md
│       └── synthetic_asset_events.csv
├── src/
│   ├── pipeline.py
│   ├── reconciliation.py
│   └── synthetic_data.py
├── tests/
├── CITATION.cff
├── CONFIDENTIALITY.md
├── LICENSE
├── REPRODUCIBILITY.md
├── requirements.txt
└── run_demo.py
```

## Run the public demonstration

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py --repetitions 3 --bootstrap 200
```

The command creates `demo_outputs/` from the synthetic dataset. For the same
perturbation and bootstrap counts used in the manuscript:

```bash
python run_demo.py --repetitions 30 --bootstrap 2000
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Interpretation

The synthetic dataset demonstrates code execution and analytical structure.
It does not reproduce the confidential industrial records or the manuscript's
numerical results.

## Data and code availability

The code, synthetic demonstration data, and documentation are public in this
repository. The original industrial dataset and all results derived from it
are withheld because of confidentiality restrictions.

## License and citation

The source code is released under the MIT License. Cite the manuscript and the
software package using `CITATION.cff`.
