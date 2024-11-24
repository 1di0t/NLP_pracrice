import tensorflow as tf


def to_dataset(sequence, len, shuffle=False, seed=None, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    dataset = dataset.window(len + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window_ds: window_ds.batch(len + 1))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100_000, seed=seed)
    dataset = dataset.batch(batch_size)
    return dataset.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)
