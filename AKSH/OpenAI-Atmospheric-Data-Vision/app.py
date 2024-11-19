from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from preprocess import preprocess_image

app = Flask(__name__)
model = tf.keras.models.load_model('models/atmospheric_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = preprocess_image(file)
    prediction = model.predict(np.expand_dims(image, axis=0))
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
