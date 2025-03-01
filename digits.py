import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

MODEL_PATH = "mnist_model.keras"


def create_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


if not os.path.exists(MODEL_PATH):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    model = create_model()
    model.fit(x_train, y_train, epochs=75, batch_size=2048, verbose=1)
    model.save(MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

image_paths = []
for root, _, files in os.walk("digits"):
    for file in files:
        if file.lower().endswith((".jpg")):
            image_paths.append(os.path.join(root, file))

counts = [0] * 10
batch_size = 4096

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i : i + batch_size]
    batch_images = []

    for path in batch_paths:
        try:
            img = Image.open(path).convert("L")
            img = img.resize((28, 28))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array.reshape(28, 28, 1)
            batch_images.append(img_array)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if batch_images:
        batch_x = np.array(batch_images)
        preds = model.predict(batch_x, verbose=0)
        digits = np.argmax(preds, axis=1)
        for d in digits:
            counts[d] += 1

print(f"Digit counts: {counts}")
