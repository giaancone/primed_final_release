"""The compression environment works end to end on a GPU: QKeras layer + tfmot pruning + one QAT step + save/reload.

Run on a GPU node: `uv run --project compression pytest -q compression/tests`. A missing GPU is a failure, not a skip:
this environment exists to train on NVIDIA A100 GPUs, and a silent CPU fallback would hide a broken CUDA install.
"""
import keras
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QDense
from qkeras.quantizers import quantized_bits
from qkeras.utils import load_qmodel

SPARSITY = 0.5


def test_gpu_visible():
    gpus = tf.config.list_physical_devices("GPU")
    print("tensorflow", tf.__version__, "keras", keras.__version__, "GPUs", gpus)
    assert gpus, "TensorFlow sees no GPU: run on a GPU node and check the nvidia-* wheels of tensorflow[and-cuda]"


def test_prune_qat_save_reload(tmp_path):
    assert tf.config.list_physical_devices("GPU"), "TensorFlow sees no GPU"
    tf.keras.utils.set_random_seed(0)
    layer = QDense(8, kernel_quantizer=quantized_bits(8, 0, alpha=1), bias_quantizer=quantized_bits(8, 0, alpha=1), name="qdense")
    model = tf.keras.Sequential([tf.keras.Input((16,)), layer])
    pruned = tfmot.sparsity.keras.prune_low_magnitude(
        model, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(SPARSITY, begin_step=0, frequency=1))
    pruned.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(64, 16)).astype("f4"), rng.normal(size=(64, 8)).astype("f4")
    pruned.fit(x, y, batch_size=64, epochs=1, verbose=0, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

    final = tfmot.sparsity.keras.strip_pruning(pruned)
    kernel = final.get_layer("qdense").kernel.numpy()
    assert np.mean(kernel == 0.0) == SPARSITY, f"sparsity {np.mean(kernel == 0.0)} != {SPARSITY}"

    path = str(tmp_path / "model.h5")
    final.save(path)
    loaded = load_qmodel(path, compile=False)
    x_fixed = rng.normal(size=(32, 16)).astype("f4")
    before, after = np.asarray(final(x_fixed, training=False)), np.asarray(loaded(x_fixed, training=False))
    assert np.mean(loaded.get_layer("qdense").kernel.numpy() == 0.0) == SPARSITY
    assert np.max(np.abs(before - after)) < 1e-6
