import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

MODEL_PATH = 'mnist_model.keras'

# Define and load the model (same as before)
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

if not os.path.exists(MODEL_PATH):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    model = create_model()
    model.fit(x_train, y_train, epochs=75, batch_size=2048, verbose=1)
    model.save(MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# Get input image path
image_path = input("Enter the path to your image: ")

# Process the image and predict
try:
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))                 # Resize to MNIST dimensions
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch dimension
    pred = model.predict(img_array, verbose=0)
    digit = np.argmax(pred, axis=1)[0]
    print(f"Predicted digit: {digit}")
except Exception as e:
    print(f"Error: {e}")