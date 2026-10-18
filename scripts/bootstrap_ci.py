"""Panel KLD mean with a 95 % bootstrap CI over conversations (the pre-registered statistic), from p4_kld.py's
kld.json (per_seq: kl_sum and positions per conversation). With two files, the paired difference B - A as well.

  python3 bootstrap_ci.py evidence/kld_panel.json [other_kld.json] [--n 10000]
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    runs = [{s["id"]: s for s in json.load(open(p))["per_seq"]} for p in [a.a] + ([a.b] if a.b else [])]
    ids = sorted(set.intersection(*[set(r) for r in runs]))
    rng = random.Random(0)

    def mean(r, sample):
        return sum(r[i]["kl_sum"] for i in sample) / sum(r[i]["positions"] for i in sample)

    draws = [[rng.choice(ids) for _ in ids] for _ in range(a.n)]
    for name, r in zip([a.a, a.b], runs):
        m = sorted(mean(r, s) for s in draws)
        print(f"{name}: KLD {mean(r, ids):.5f}, 95% CI [{m[int(0.025 * a.n)]:.5f}, {m[int(0.975 * a.n)]:.5f}], "
              f"{len(ids)} conversations")
    if a.b:
        d = sorted(mean(runs[1], s) - mean(runs[0], s) for s in draws)
        print(f"B - A: {mean(runs[1], ids) - mean(runs[0], ids):+.5f}, 95% CI [{d[int(0.025 * a.n)]:+.5f}, "
              f"{d[int(0.975 * a.n)]:+.5f}]")


if __name__ == "__main__":
    main()
