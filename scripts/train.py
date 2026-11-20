"""Run the stages of one config. Usage: python scripts/train.py configs/paper/dro.yaml run.name=x seed=1 ...
Exits with the trainer's PREEMPTED code when a stage stopped at its deadline or on SIGTERM (start it again to resume)."""
import sys

from wavefm.train import pipeline
from wavefm.utils import config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    result = pipeline.run(config.apply_overrides(config.load(sys.argv[1]), sys.argv[2:]))
    if result["preempted"]:
        from wavefm.train.trainer import PREEMPTED
        sys.exit(PREEMPTED)
