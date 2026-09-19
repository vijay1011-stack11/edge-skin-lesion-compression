"""
Benchmark every model variant in models/: accuracy, per-class F1, critical-class F1,
model size, and single-core CPU latency. Writes the Pareto-frontier table the report
is built around.

Usage:
    python -m src.benchmark
"""
import os
import glob
import time
import json
import numpy as np
import tensorflow as tf
import pandas as pd

from . import config, data, evaluate


def _prep_test_arrays(test_df):
    imgs, labels = [], []
    for _, row in test_df.iterrows():
        img = tf.io.read_file(row["path"])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        imgs.append(img.numpy())
        labels.append(row["label"])
    return np.array(imgs, dtype=np.float32), np.array(labels)


def benchmark_tflite(tflite_path, test_imgs, test_labels, num_runs=config.BENCHMARK_NUM_RUNS):
    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=config.BENCHMARK_THREADS)
    interpreter.allocate_tensors()

    report = evaluate.evaluate_tflite_model(interpreter, test_imgs, test_labels)

    # Latency: average over num_runs single-image inferences (warm-up excluded).
    input_details = interpreter.get_input_details()[0]
    sample = test_imgs[0:1]
    if input_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = input_details["quantization"]
        sample = (sample / scale + zero_point).astype(input_details["dtype"])
    else:
        sample = sample.astype(np.float32)

    for _ in range(5):  # warm-up
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()

    start = time.perf_counter()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
    elapsed = time.perf_counter() - start
    avg_latency_ms = (elapsed / num_runs) * 1000

    size_kb = os.path.getsize(tflite_path) / 1024
    report["latency_ms"] = avg_latency_ms
    report["size_kb"] = size_kb
    report["model"] = os.path.basename(tflite_path)
    return report


def main():
    df = data.load_metadata()
    _, _, test_df = data.split_by_lesion(df)
    test_imgs, test_labels = _prep_test_arrays(test_df)

    tflite_files = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*.tflite")))
    if not tflite_files:
        raise FileNotFoundError("No .tflite files found. Run compress.py first.")

    rows = []
    for path in tflite_files:
        print(f"Benchmarking {os.path.basename(path)}...")
        report = benchmark_tflite(path, test_imgs, test_labels)
        rows.append({
            "model": report["model"],
            "size_kb": round(report["size_kb"], 1),
            "latency_ms": round(report["latency_ms"], 3),
            "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(report["macro_f1"], 4),
            **{f"f1_{c}": round(v, 4) for c, v in report["per_class_f1"].items()},
        })

    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(config.REPORTS_DIR, "pareto_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved Pareto-frontier table -> {out_csv}\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
