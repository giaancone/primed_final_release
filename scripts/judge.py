"""Judge the curves of a stage (analysis/JUDGING.md).

Usage: python scripts/judge.py --pending <output>      figures (<stage>/curves.png) and table of every stage to judge
       python scripts/judge.py <stage dir> <pass|overfit|undertrained|unclear> "<reason>" --judge <who>
"""
import argparse

from analysis import judge
from wavefm.train import rundir

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pending", metavar="OUTPUT")
    ap.add_argument("stage", nargs="?"); ap.add_argument("verdict", nargs="?", choices=rundir.VERDICTS)
    ap.add_argument("reason", nargs="?"); ap.add_argument("--judge")
    a = ap.parse_args()
    if a.pending:
        print(judge.show(judge.prepare_pending(a.pending)))
    elif a.stage and a.verdict and a.reason and a.judge:
        print(rundir.write_verdict(a.stage, a.verdict, a.reason, a.judge))
    else:
        ap.error("give --pending <output>, or a stage directory, a verdict, a reason and --judge")
