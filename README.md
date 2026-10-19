# HSMI: scoring data and robustness analysis

Data and code supporting the manuscript "Hydrogen Refueling Station Safety Maturity Assessment: An Integrated Incident, Regulatory, and Multi-Criteria Decision Framework".

## Contents

| Path | Description |
|---|---|
| `data/hsmi_scores.csv` | Scores (0 to 10) of five jurisdictions on ten parameters, with the baseline weights |
| `data/HSMI_scoring_workbook.xlsx` | Same scores with the SAW, TOPSIS and sensitivity results as in the manuscript |
| `hsmi_montecarlo.py` | Monte Carlo rank-acceptability analysis |
| `results/rank_acceptability.csv` | Probability of each jurisdiction holding each rank |
| `results/run_log.txt` | Console output of the run reported in the manuscript |

## Method

Weights are drawn from a Dirichlet distribution whose mean equals the baseline weights, so every draw sums to one. Scores are shifted by a uniform random amount and clipped to the 0 to 10 range. Each draw is ranked with simple additive weighting (SAW) and with TOPSIS using vector normalisation. Two scenarios are run with 10,000 draws each:

| Scenario | Dirichlet concentration | Score shift |
|---|---|---|
| moderate | 100 | +/- 1 point |
| wide | 30 | +/- 2 points |

A higher concentration keeps the sampled weights closer to the baseline. The random seed is fixed (20261015), so results are exactly reproducible.

## Usage

```
pip install -r requirements.txt
python hsmi_montecarlo.py
```

Options: `--data`, `--out`, `--draws`, `--seed`.

## Notes

The scores are ordinal expert judgments based on the regulatory and engineering evidence discussed in the manuscript. The weights are informed by root-cause frequencies in the JRC Hydrogen Incidents and Accidents Database (extract of October 2026) and are not statistically estimated. The analysis tests how much the ranking depends on these judgments. It does not validate them.

## Licence

Code and data are released under the MIT licence (see `LICENSE`).
