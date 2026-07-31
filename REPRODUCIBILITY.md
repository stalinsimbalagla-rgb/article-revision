# Reproducibility record

## Public demonstration

`run_demo.py` executes the same public functions on synthetic data. Its
outputs are expected to differ from the manuscript values because the
industrial row-level data are withheld. Reproducibility here means:

- the target-construction and validation logic is inspectable;
- the analysis executes deterministically with fixed seeds;
- the confidentiality boundary is explicit.

The manuscript analysis used 30 repetitions for each combination of
hypothetical relabeling rate and direction. The public demonstration uses
three repetitions by default to reduce execution time; pass
`--repetitions 30 --bootstrap 2000` to exercise the full computational
settings on synthetic data.
