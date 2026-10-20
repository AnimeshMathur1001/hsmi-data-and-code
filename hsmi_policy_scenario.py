"""Policy scenario for India: SAW score and TOPSIS closeness coefficient.

Parameters P1, P7 and P8 are set to the low and high values of the policy
scenario in the manuscript (P1 = 5 or 6, P7 = 6 or 7, P8 = 7 or 8). Run from
the repository root: python hsmi_policy_scenario.py
"""
import csv
import numpy as np
from hsmi_montecarlo import load, saw, topsis, ranks

SCENARIOS = {"baseline": None, "low": (5, 6, 7), "high": (6, 7, 8)}


def main():
    names, scores, weights = load("data/hsmi_scores.csv")
    india = names.index("India")
    rows = []
    for tag, vals in SCENARIOS.items():
        s = scores.copy()
        if vals:
            s[0, india], s[6, india], s[7, india] = vals
        saw_v, top_v = saw(s, weights), topsis(s, weights)
        rows.append({
            "scenario": tag,
            "India_SAW": round(float(saw_v[india]), 3),
            "India_TOPSIS_CC": round(float(top_v[india]), 3),
            "India_SAW_rank": int(ranks(saw_v)[india]),
            "India_TOPSIS_rank": int(ranks(top_v)[india]),
        })
    with open("results/policy_scenario.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
