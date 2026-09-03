import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "prepared")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "catneural.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    labels="inferred",
    label_mode="binary",
    class_names=["not_cat", "cat"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    labels="inferred",
    label_mode="binary",
    class_names=["not_cat", "cat"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
], name="augmentation")

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)],
)

model.save(MODEL_PATH)
print(f"Modelo salvo em: {MODEL_PATH}")
