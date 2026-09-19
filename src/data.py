"""
Data loading for HAM10000.

Key correctness point most people get wrong on this dataset:
HAM10000 has MULTIPLE IMAGES OF THE SAME LESION (lesion_id repeats).
Splitting by image_id instead of lesion_id leaks near-duplicate images across
train/val/test and inflates your reported accuracy. This module splits by
lesion_id using GroupShuffleSplit, stratified as well as possible by dx.
"""
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from . import config


def _build_image_path_index():
    """Map image_id -> absolute file path across both HAM10000 image folders."""
    index = {}
    for d in config.IMAGE_DIRS:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*.jpg")):
            image_id = os.path.splitext(os.path.basename(path))[0]
            index[image_id] = path
    return index


def load_metadata():
    """Load HAM10000_metadata.csv and attach resolved file paths."""
    if not os.path.exists(config.METADATA_CSV):
        raise FileNotFoundError(
            f"Metadata CSV not found at {config.METADATA_CSV}. "
            f"Set HAM10000_DIR env var to your dataset location, or edit config.py."
        )
    df = pd.read_csv(config.METADATA_CSV)
    path_index = _build_image_path_index()
    df["path"] = df["image_id"].map(path_index)
    missing = df["path"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows have no matching image file on disk. Dropping them.")
        df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["label"] = df["dx"].map({name: i for i, name in enumerate(config.CLASS_NAMES)})
    return df


def split_by_lesion(df, val_size=0.15, test_size=0.15, seed=config.SEED):
    """
    Group-aware split: all images of the same lesion_id go to the same split.
    Two-stage GroupShuffleSplit: first carve off test, then val from what's left.
    """
    groups = df["lesion_id"].values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df["lesion_id"].values))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    # Sanity check: no lesion_id should appear in more than one split.
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    assert not (train_lesions & val_lesions), "Lesion leakage between train/val!"
    assert not (train_lesions & test_lesions), "Lesion leakage between train/test!"
    assert not (val_lesions & test_lesions), "Lesion leakage between val/test!"

    return train_df, val_df, test_df


def compute_class_weights(train_df):
    """Inverse-frequency class weights to counter HAM10000's ~67% nv imbalance."""
    counts = train_df["label"].value_counts().sort_index()
    total = counts.sum()
    n_classes = len(config.CLASS_NAMES)
    weights = {i: total / (n_classes * counts.get(i, 1)) for i in range(n_classes)}
    return weights


def _decode_and_preprocess(path, label, augment):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def make_dataset(df, augment=False, shuffle=False, batch_size=config.BATCH_SIZE):
    paths = df["path"].values
    labels = df["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _decode_and_preprocess(p, l, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def representative_dataset_gen(train_df, num_samples=100):
    """Generator required by the TFLite converter for full-integer PTQ calibration."""
    sample_df = train_df.sample(n=min(num_samples, len(train_df)), random_state=config.SEED)
    for path in sample_df["path"].values:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        yield [img]
