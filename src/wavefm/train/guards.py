"""Run-time checks that turn silent failures into errors. Each one is exercised by tests/test_faults.py."""
import os
import socket
import threading
import time
from pathlib import Path

import torch

from wavefm.train import rundir


class GuardError(RuntimeError):
    pass


# --- one process per stage directory ---------------------------------------------------------------------------------
class RunLock:
    """``with RunLock(stage_dir):`` -- a lock file with a heartbeat. A second process (any node) refuses to start
    while the heartbeat is fresh; a lock older than ``stale_s`` belongs to a dead process (killed node) and is taken
    over.

    Acquisition is atomic: the file is created with O_CREAT | O_EXCL, so of two processes exactly one gets it. A stale
    lock is first renamed to a name of this process (atomic too) and removed only if it is still stale after the
    rename; a fresh one caught by a race is put back with a hard link, which never overwrites.
    The heartbeat is a daemon thread that touches the file every ``stale_s / 4`` seconds, so a long evaluation or
    compilation never lets the lock look stale. If another process took the lock over (its owner is in the file),
    ``check()`` raises; the trainer calls it at every evaluation and checkpoint.
    """

    def __init__(self, run_dir, stale_s: float = 900):
        self.path, self.stale_s, self.interval = Path(run_dir) / "run.lock", float(stale_s), float(stale_s) / 4
        if not 0 < self.interval < self.stale_s:
            raise GuardError(f"RunLock: heartbeat interval {self.interval} s must be shorter than the staleness "
                             f"{stale_s} s")
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{os.environ.get('SLURM_JOB_ID', '-')}"
        self.lost, self._stop, self._thread = None, threading.Event(), None

    def _read(self, path=None):
        try:
            return Path(path or self.path).read_text().strip()
        except FileNotFoundError:
            return None

    def _take_stale(self):
        """Remove the lock file if it is stale; leave it if it is fresh."""
        try:
            if time.time() - self.path.stat().st_mtime < self.stale_s:
                raise GuardError(f"{self.path.parent} is being trained by {self._read()} (lock {self.path}): wait, or "
                                 f"use another run.name")
        except FileNotFoundError:
            return
        grabbed = self.path.with_name(f".{self.path.name}.stale.{os.getpid()}")
        try:
            os.rename(self.path, grabbed)
        except FileNotFoundError:
            return
        if time.time() - grabbed.stat().st_mtime < self.stale_s:     # another process took it over meanwhile
            try:
                os.link(grabbed, self.path)
            except FileExistsError:
                pass
            grabbed.unlink()
            raise GuardError(f"{self.path.parent} was just taken by {self._read()}")
        print(f"[lock] taking over the stale lock of {self._read(grabbed)} ({self.path})", flush=True)
        grabbed.unlink()

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._take_stale()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            raise GuardError(f"{self.path.parent} is being trained by {self._read()}: wait, or use another "
                             f"run.name") from None
        with os.fdopen(fd, "w") as f:
            f.write(self.owner)
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, daemon=True, name="RunLock heartbeat")
        self._thread.start()
        return self

    def _beat(self):
        while not self._stop.wait(self.interval):
            who = self._read()
            if who != self.owner:
                self.lost = who
                return
            try:
                os.utime(self.path)
            except FileNotFoundError:
                self.lost = None
                return

    def check(self):
        """Raise if this process no longer owns the lock."""
        if self.lost is not None or self._read() != self.owner:
            raise GuardError(f"{self.path} was taken over by {self.lost or self._read()}: two processes trained the "
                             f"same stage; this one stops")

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._read() == self.owner:
            self.path.unlink()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# --- the files of a finished stage ---------------------------------------------------------------------------------
def check_artifact(out, result: dict, name: str):
    """Before a ``predict`` reads ``<out>/<name>``: its sha256 must be the one the stage recorded in result.json
    (``result["artifacts"]``), else the file was edited or replaced after the stage finished."""
    path = Path(out) / name
    want = result["artifacts"].get(name, {}).get("sha256")
    if want is None or not path.is_file():
        raise GuardError(f"{path}: the stage {out} did not promise this file or it is gone: run the stage again")
    if rundir.file_sha256(path) != want:
        raise GuardError(f"{path} changed after the stage finished (sha256 differs from its result.json): run the "
                         f"stage again; its predictions would not be those of the recorded result")


# --- numerics ---------------------------------------------------------------------------------------------------------
def check_strict_fp32(*tensors_or_modules):
    """Strict fp32: TF32 off for matmul and cudnn, float32 matmul precision "highest", no autocast on any device,
    and every parameter / tensor given is float32. Called by the trainer before the first step."""
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"):
        raise GuardError("TF32 is on (torch.backends.cuda.matmul / cudnn allow_tf32, or the float32 matmul precision "
                         "is not 'highest'): networks train in strict fp32; call wavefm.utils.runtime.setup_numerics()")
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise GuardError("autocast is active: networks train in strict fp32; do not call the stage inside autocast")
    for obj in tensors_or_modules:
        named = obj.named_parameters() if isinstance(obj, torch.nn.Module) else obj.items()
        bad = [n for n, p in named if p.dtype != torch.float32]
        if bad:
            raise GuardError(f"{len(bad)} parameters are not float32, e.g. {bad[0]}: networks train in strict fp32")


def check_gradients(names: dict, groups: list, unused: tuple = ()):
    """After the first backward pass, tensor by tensor: nothing in a group with a positive learning rate is frozen, and
    every trained tensor is connected to the loss (has a gradient; zero is allowed, e.g. the layer before a
    zero-initialised adapter output). ``names``: id(tensor) -> name. ``unused`` lists name fragments of tensors that
    legitimately get none (TimesFM 3.0 single-channel shortcut)."""
    for g in groups:
        if g["lr"] <= 0:
            continue
        for p in g["params"]:
            name = names.get(id(p), "<unnamed tensor>")
            if any(u in name for u in unused):
                continue
            if not p.requires_grad:
                raise GuardError(f"{name} is frozen but sits in the trained group {g.get('name', '?')}")
            if p.grad is None:
                raise GuardError(f"{name} gets no gradient: disconnected from the loss")
