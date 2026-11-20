#!/bin/bash
# Set your account with sbatch -A <account>; adapt the partition and constraint to your cluster.
# An example SLURM job: one pipeline on one GPU. Adapt it to your own way of submitting. Submit from the repository root:
#     sbatch scripts/example_job.sh <config> [key=value ..]
# e.g. sbatch -t 0:15:00 scripts/example_job.sh configs/smoke/dro.yaml run.group=smoke run.name=job
# On SIGTERM (120 s before the time limit) the running stage checkpoints and the job exits with code 3: submit the
# same command again and it resumes. compress runs afterwards in the other environment (the pipeline prints how).
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus 1
#SBATCH -t 4:00:00
#SBATCH -J wavefm
#SBATCH -o %x-%j.out
#SBATCH --signal=B:TERM@120
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source setup.sh
exec python scripts/train.py "$@"
