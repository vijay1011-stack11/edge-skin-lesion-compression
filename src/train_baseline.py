"""
Train the MobileNetV2 baseline on HAM10000.
Two-phase transfer learning: frozen backbone, then fine-tune top layers.

Usage:
    python -m src.train_baseline
"""
import os
import numpy as np
import tensorflow as tf

from . import config, data, model as model_lib, evaluate


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    print("Loading metadata...")
    df = data.load_metadata()
    train_df, val_df, test_df = data.split_by_lesion(df)
    print(f"Train: {len(train_df)} images / {train_df['lesion_id'].nunique()} lesions")
    print(f"Val:   {len(val_df)} images / {val_df['lesion_id'].nunique()} lesions")
    print(f"Test:  {len(test_df)} images / {test_df['lesion_id'].nunique()} lesions")

    class_weights = data.compute_class_weights(train_df)
    print("Class weights (inverse frequency):", class_weights)

    train_ds = data.make_dataset(train_df, augment=True, shuffle=True)
    val_ds = data.make_dataset(val_df, augment=False, shuffle=False)
    test_ds = data.make_dataset(test_df, augment=False, shuffle=False)

    print("Building model (phase 1: frozen backbone)...")
    model, base = model_lib.build_mobilenetv2(freeze_base=True)
    model_lib.compile_model(model, lr=1e-3)
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.BASELINE_EPOCHS_FROZEN,
        class_weight=class_weights,
    )

    print("Phase 2: unfreezing top layers, fine-tuning...")
    model_lib.unfreeze_top_layers(base)
    model_lib.compile_model(model, lr=1e-5)

    ckpt_path = os.path.join(config.OUTPUT_DIR, "baseline_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.BASELINE_EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    final_path = os.path.join(config.OUTPUT_DIR, "baseline_final.keras")
    model.save(final_path)
    print(f"Saved baseline model -> {final_path}")

    print("Evaluating baseline on held-out test set...")
    true_labels = test_df["label"].values
    report = evaluate.evaluate_keras_model(model, test_ds, true_labels)
    evaluate.save_report(report, os.path.join(config.REPORTS_DIR, "baseline_eval.json"))
    print(f"Baseline accuracy: {report['accuracy']:.4f}  macro-F1: {report['macro_f1']:.4f}")
    print(f"Critical-class F1: {report['critical_class_f1']}")


if __name__ == "__main__":
    main()
