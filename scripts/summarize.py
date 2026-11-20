"""The summary of one stage over the runs under an output directory: accepted runs only, mean and spread over the seeds
of each recipe (analysis/summary.py).

Usage: python scripts/summarize.py <output> <stage> [--seeds 0,1,2] [--efficiency <metric> <figure.png>]
"""
import argparse

from analysis import judge, read, summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("output"); ap.add_argument("stage")
    ap.add_argument("--seeds", help="the seeds every set must have, e.g. 0,1,2")
    ap.add_argument("--efficiency", nargs=2, metavar=("METRIC", "FIGURE"))
    a = ap.parse_args()
    runs = read.runs(a.output)
    for r, why in summary.rejected(runs, a.stage):
        print(f"left out {r['group']}/{r['name']}: {why}")
    chosen = summary.select(runs, a.stage)
    sets = summary.groups(chosen, a.stage)
    if a.seeds:
        sets = summary.require_seeds(sets, [int(s) for s in a.seeds.split(",")])
    rows = summary.table(sets, a.stage)
    print(judge.show(rows, [k for k in rows[0] if k != "data_hash"]) if rows else "no accepted runs")
    if a.efficiency:
        from analysis.efficiency import efficiency
        efficiency([r for g in sets for r in g["runs"]], *a.efficiency)
