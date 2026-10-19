"""Monte Carlo robustness analysis for the HSMI ranking.

Weights are sampled from a Dirichlet distribution centred on the baseline
weights. Scores are perturbed uniformly by +/- delta and clipped to [0, 10].
Each draw is ranked with SAW and with TOPSIS (vector normalisation).
"""
import argparse
import csv
import numpy as np

SEED = 20261015
N_DRAWS = 10000


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    names = [k for k in rows[0] if k not in ("parameter", "weight")]
    scores = np.array([[float(r[n]) for n in names] for r in rows])
    weights = np.array([float(r["weight"]) for r in rows])
    return names, scores, weights


def saw(scores, weights):
    return weights @ scores


def topsis(scores, weights):
    norm = scores / np.sqrt((scores ** 2).sum(axis=1, keepdims=True))
    v = norm * weights[:, None]
    d_plus = np.sqrt(((v - v.max(axis=1, keepdims=True)) ** 2).sum(axis=0))
    d_minus = np.sqrt(((v - v.min(axis=1, keepdims=True)) ** 2).sum(axis=0))
    return d_minus / (d_plus + d_minus)


def ranks(values):
    order = np.argsort(-values)
    r = np.empty(len(values), dtype=int)
    r[order] = np.arange(1, len(values) + 1)
    return r


def simulate(scores, weights, concentration, delta, n, rng):
    m = scores.shape[1]
    out = {"SAW": np.zeros((m, m), int), "TOPSIS": np.zeros((m, m), int)}
    for _ in range(n):
        w = rng.dirichlet(concentration * weights)
        s = np.clip(scores + rng.uniform(-delta, delta, scores.shape), 0, 10)
        for key, fn in (("SAW", saw), ("TOPSIS", topsis)):
            r = ranks(fn(s, w))
            out[key][np.arange(m), r - 1] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hsmi_scores.csv")
    p.add_argument("--out", default="results/rank_acceptability.csv")
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()

    names, scores, weights = load(a.data)
    assert abs(weights.sum() - 1.0) < 1e-9

    print("Baseline SAW   ", {n: round(float(v), 2) for n, v in zip(names, saw(scores, weights))})
    print("Baseline TOPSIS", {n: round(float(v), 3) for n, v in zip(names, topsis(scores, weights))})

    scenarios = [("moderate", 100, 1.0), ("wide", 30, 2.0)]
    rows = []
    for label, conc, delta in scenarios:
        rng = np.random.default_rng(a.seed)
        res = simulate(scores, weights, conc, delta, a.draws, rng)
        for method, counts in res.items():
            for i, n in enumerate(names):
                for k in range(len(names)):
                    rows.append([label, conc, delta, method, n, k + 1,
                                 round(counts[i, k] / a.draws, 4)])
            print(f"\n{label} (concentration {conc}, score shift +/-{delta}), {method}")
            print("rank:        " + "".join(f"{k + 1:>8}" for k in range(len(names))))
            for i, n in enumerate(names):
                print(f"{n:<12}" + "".join(f"{counts[i, k] / a.draws:8.3f}" for k in range(len(names))))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "dirichlet_concentration", "score_shift", "method",
                    "jurisdiction", "rank", "probability"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
 
