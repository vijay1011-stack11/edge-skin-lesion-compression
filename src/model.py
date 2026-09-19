"""MobileNetV2 transfer-learning model for HAM10000 (7-class)."""
import tensorflow as tf
from . import config


def build_mobilenetv2(num_classes=config.NUM_CLASSES, freeze_base=True):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights=config.MOBILENET_WEIGHTS,
    )
    base.trainable = not freeze_base

    # IMPORTANT: build off base.input/base.output directly rather than calling
    # base(inputs) as a layer. Calling it as a layer nests the whole MobileNetV2
    # graph as a single sub-model inside our Model, which quietly breaks
    # tfmot's quantize_model later ("Quantizing a keras Model inside another
    # keras Model is not supported"). Tracing through base.input/base.output
    # flattens all of MobileNetV2's layers directly into this model's graph,
    # which is what tfmot (and QAT conversion) needs. Freezing still works the
    # same way via base.trainable / individual layer.trainable, since these are
    # the same underlying layer objects either way -- and Keras special-cases
    # BatchNormalization to run in inference mode whenever layer.trainable is
    # False, regardless of the outer model's training flag.
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs, name="mobilenetv2_ham10000")
    return model, base


def unfreeze_top_layers(base, from_index=config.FINETUNE_UNFREEZE_FROM):
    base.trainable = True
    for layer in base.layers[:from_index]:
        layer.trainable = False
    return base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
