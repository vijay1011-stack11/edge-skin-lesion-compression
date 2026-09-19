"""Per-class evaluation — the core differentiator of this project.
Aggregate accuracy hides class-level collapse; everything here reports per-class too."""
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config


def evaluate_keras_model(model, dataset, true_labels):
    probs = model.predict(dataset, verbose=0)
    preds = np.argmax(probs, axis=1)
    return _report(true_labels, preds)


def evaluate_tflite_model(interpreter, dataset_as_numpy, true_labels):
    """dataset_as_numpy: list/array of preprocessed images (float32, NHWC)."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = []
    for img in dataset_as_numpy:
        img = np.expand_dims(img, axis=0)
        if input_details["dtype"] == np.uint8 or input_details["dtype"] == np.int8:
            scale, zero_point = input_details["quantization"]
            img = (img / scale + zero_point).astype(input_details["dtype"])
        else:
            img = img.astype(np.float32)
        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        preds.append(int(np.argmax(out)))
    return _report(true_labels, np.array(preds))


def _report(true_labels, preds):
    all_labels = list(range(config.NUM_CLASSES))
    report_dict = classification_report(
        true_labels, preds, labels=all_labels, target_names=config.CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(true_labels, preds, labels=all_labels)
    critical_f1 = {
        c: report_dict[c]["f1-score"] for c in config.CRITICAL_CLASSES if c in report_dict
    }
    return {
        "accuracy": report_dict["accuracy"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "per_class_f1": {c: report_dict[c]["f1-score"] for c in config.CLASS_NAMES if c in report_dict},
        "critical_class_f1": critical_f1,
        "confusion_matrix": cm.tolist(),
    }


def save_report(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved eval report -> {path}")
