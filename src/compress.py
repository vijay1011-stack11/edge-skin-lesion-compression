"""
Compression pipeline: structured pruning, INT8 post-training quantization (PTQ),
quantization-aware training (QAT), and pruning+PTQ combined.

Produces one TFLite file per configuration in models/, named so benchmark.py can
sweep over all of them automatically.

Usage:
    python -m src.compress
"""
import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from . import config, data, model as model_lib


def _convert_to_tflite(keras_model, out_path, representative_gen=None, full_int8=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_gen is not None:
        converter.representative_dataset = representative_gen
    if full_int8:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")
    return out_path


def run_ptq(baseline_path, train_df):
    """Dynamic-range PTQ (no calibration) and full-integer PTQ (with calibration)."""
    model = tf.keras.models.load_model(baseline_path)

    dynamic_path = os.path.join(config.OUTPUT_DIR, "ptq_dynamic.tflite")
    _convert_to_tflite(model, dynamic_path)

    int8_path = os.path.join(config.OUTPUT_DIR, "ptq_int8.tflite")
    rep_gen = lambda: data.representative_dataset_gen(train_df)
    _convert_to_tflite(model, int8_path, representative_gen=rep_gen, full_int8=True)

    return {"ptq_dynamic": dynamic_path, "ptq_int8": int8_path}


def run_pruning(baseline_path, train_df, val_df, sparsity):
    """Structured magnitude pruning at a target sparsity, then fine-tune to recover accuracy."""
    model = tf.keras.models.load_model(baseline_path)

    train_ds = data.make_dataset(train_df, augment=True, shuffle=True)
    val_ds = data.make_dataset(val_df, augment=False, shuffle=False)
    class_weights = data.compute_class_weights(train_df)

    num_train_batches = len(train_df) // config.BATCH_SIZE
    end_step = num_train_batches * config.PRUNE_EPOCHS

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=sparsity,
            begin_step=0, end_step=end_step,
        )
    }
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_lib.compile_model(pruned_model, lr=1e-5)

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(
        train_ds, validation_data=val_ds, epochs=config.PRUNE_EPOCHS,
        class_weight=class_weights, callbacks=callbacks,
    )

    stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
    keras_out = os.path.join(config.OUTPUT_DIR, f"pruned_{int(sparsity*100)}.keras")
    stripped.save(keras_out)

    tflite_out = os.path.join(config.OUTPUT_DIR, f"pruned_{int(sparsity*100)}.tflite")
    _convert_to_tflite(stripped, tflite_out)

    return keras_out, tflite_out


def run_pruning_plus_int8(pruned_keras_path, train_df, sparsity):
    """Combined: take an already-pruned model and additionally apply full-int8 PTQ."""
    model = tf.keras.models.load_model(pruned_keras_path)
    out_path = os.path.join(config.OUTPUT_DIR, f"pruned_{int(sparsity*100)}_int8.tflite")
    rep_gen = lambda: data.representative_dataset_gen(train_df)
    _convert_to_tflite(model, out_path, representative_gen=rep_gen, full_int8=True)
    return out_path


def run_qat(baseline_path, train_df, val_df):
    """Quantization-aware training: simulates int8 quantization during fine-tuning."""
    model = tf.keras.models.load_model(baseline_path)
    quant_aware_model = tfmot.quantization.keras.quantize_model(model)
    model_lib.compile_model(quant_aware_model, lr=1e-5)

    train_ds = data.make_dataset(train_df, augment=True, shuffle=True)
    val_ds = data.make_dataset(val_df, augment=False, shuffle=False)
    class_weights = data.compute_class_weights(train_df)

    quant_aware_model.fit(
        train_ds, validation_data=val_ds, epochs=config.QAT_EPOCHS,
        class_weight=class_weights,
    )

    tflite_out = os.path.join(config.OUTPUT_DIR, "qat_int8.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_out, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {tflite_out} ({os.path.getsize(tflite_out)/1024:.1f} KB)")
    return tflite_out


def main():
    baseline_path = os.path.join(config.OUTPUT_DIR, "baseline_final.keras")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError("Run train_baseline.py first.")

    df = data.load_metadata()
    train_df, val_df, test_df = data.split_by_lesion(df)

    print("=== PTQ ===")
    run_ptq(baseline_path, train_df)

    print("=== Pruning sweep ===")
    for sparsity in config.PRUNING_TARGET_SPARSITY:
        keras_out, _ = run_pruning(baseline_path, train_df, val_df, sparsity)
        print(f"=== Pruning({sparsity}) + INT8 PTQ ===")
        run_pruning_plus_int8(keras_out, train_df, sparsity)

    print("=== QAT ===")
    run_qat(baseline_path, train_df, val_df)

    print("Done. Run: python -m src.benchmark")


if __name__ == "__main__":
    main()
