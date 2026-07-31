# Synthetic demonstration data

`synthetic_asset_events.csv` is generated deterministically by
`src/synthetic_data.py`.

- Asset identifiers follow the `SYN-####` pattern.
- Equipment names and process groups are generic.
- Dates and measurements are simulated.
- No row was copied, transformed, or sampled from the confidential dataset.
- Numerical outputs from this file must not be compared with the manuscript
  as if they were a replication of the industrial results.

Regenerate the file with:

```bash
python -m src.synthetic_data
```
