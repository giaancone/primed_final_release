"""The overview figure of a data set: a few example waveforms of ``dev`` with their labels, and the distribution of
every label column of ``dev``.

Only ``dev`` is read (through data.store.Split): a figure that looked at ``heldout`` would be a look at the answers.
Names, units, the time axis and the waveform unit all come from the split's meta.json.
"""
from pathlib import Path

import numpy as np

from analysis.curves import _figure, _save
from data import store


def overview(set_name, config, path, n_examples=4, root=store.ROOT):
    """Draw the overview of ``data/<set>/configs/<config>.yaml`` to ``path`` (PNG). Returns the example positions."""
    dev = store.Split(Path(store.load_config(set_name, config, root)["out"]) / "dev")
    m = dev.meta
    pos = np.linspace(0, dev.n - 1, n_examples).round().astype(int)
    t = m["t_first_ns"] + m["dt_ns"] * np.arange(dev.length)
    plt = _figure()
    k = len(dev.names)
    fig = plt.figure(figsize=(max(8, 3.2 * k), 7))
    ax = fig.add_subplot(2, 1, 1)
    for i, w in zip(pos, dev.take(pos)):
        lab = ", ".join(f"{s['name']} = {v:.3g} {s['unit']}" for s, v in zip(m["labels"], dev.labels[i]))
        ax.plot(t, w, lw=0.8, label=f"row {dev.rows[i]}: {lab}")
    ax.set_xlabel("time [ns]"); ax.set_ylabel(f"amplitude [{m['unit']}]")
    ax.set_title(f"{m['set']} / {m['config']}, dev ({dev.n} events)", fontsize=11)
    ax.legend(frameon=False, fontsize=7)
    for j, s in enumerate(m["labels"]):
        a = fig.add_subplot(2, k, k + j + 1)
        a.hist(dev.labels[:, j], bins=50, color="0.4")
        a.set_xlabel(f"{s['name']} [{s['unit']}]")
    fig.tight_layout(); _save(fig, path); plt.close(fig)
    return pos
