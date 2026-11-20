"""dch_wf: count the primary ionisation clusters in simulated CEPC drift-chamber waveforms (see README.md).

``build`` writes the splits (labels, classes, digests; the waveforms stay in the source npz files), ``evaluate`` is
the one metric definition. Both are reached through data.store: ``store.prepare(store.load_config("dch_wf", name))``
and ``store.evaluate("dch_wf")``.
"""
