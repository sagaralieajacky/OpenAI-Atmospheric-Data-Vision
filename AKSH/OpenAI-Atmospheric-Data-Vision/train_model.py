import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_model():
    """Builds a Convolutional Neural Network."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes as an example
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_data, train_labels):
    """Trains the model with given data."""
    model.fit(train_data, train_labels, epochs=10, batch_size=32)
    model.save('models/atmospheric_model.h5')

# Example usage
if __name__ == "__main__":
    # Replace with real training data
    train_data = np.random.rand(100, 256, 256, 3)
    train_labels = np.random.randint(0, 10, size=100)
    
    model = build_model()
    train_model(model, train_data, train_labels)
    print("Model trained and saved.")
