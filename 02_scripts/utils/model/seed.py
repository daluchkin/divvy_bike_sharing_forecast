import os
import random
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


def set_seed(SEED: int = 42, verbose=True):
    """
    Sets seeds across Python, NumPy, and TensorFlow for reproducibility.

    Parameters:
        SEED : int
            Random seed to set globally.
    """
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # TensorFlow
    if tf is not None:
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.random.set_seed(SEED)
        try:
            tf.keras.utils.set_random_seed(SEED)
        except AttributeError:
            pass

    if verbose:
        print(f"Seed {SEED} has been set.")