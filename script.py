import os
import json
import numpy as np
import pickle
import librosa
from tensorflow.keras.models import model_from_json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the path to the model files within the container
model_dir = "model"

# Load the model
def load_model():
    json_file_path = os.path.join(model_dir, "CNN_model.json")
    weights_file_path = os.path.join(model_dir, "CNN_model_weights.h5")

    json_file = open(json_file_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_file_path)

    # Compile the model (necessary for making predictions)
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return loaded_model

# Load the scaler and encoder
def load_objects():
    scaler_path = os.path.join(model_dir, "scaler2.pickle")
    encoder_path = os.path.join(model_dir, "encoder2.pickle")

    with open(scaler_path, 'rb') as f:
        scaler2 = pickle.load(f)

    with open(encoder_path, 'rb') as f:
        encoder2 = pickle.load(f)

    return scaler2, encoder2

# Feature extraction and preprocessing
def zcr(data, frame_length, hop_length):
    zcr_result = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_result)

def rms(data, frame_length=2048, hop_length=512):
    rmse_result = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_result)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_result = librosa.feature.mfcc(y=data, sr=sr)
    return np.ravel(mfcc_result.T) if flatten else mfcc_result.T

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.hstack((zcr(data, frame_length, hop_length),
                        rms(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result

def get_predict_feat(path):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    result = extract_features(data, sr)
    result = np.array(result)
    result = np.reshape(result, newshape=(1, 2376))
    
    scaler2, _ = load_objects()
    input_result = scaler2.transform(result)
    final_result = np.expand_dims(input_result, axis=2)

    return final_result

def prediction(path):
    res = get_predict_feat(path)
    predictions = model.predict(res)
    _, encoder2 = load_objects()
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the file from the request
    file = request.files["file"]
    file_path = "temp_audio.wav"
    file.save(file_path)  # Save the file temporarily

    emotion = prediction(file_path, model)

    os.remove(file_path)  # Remove the temporary file

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    model = load_model()  # Load the model once

    # Heroku provides the port dynamically, so we use the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    
    # We bind to "0.0.0.0" to allow external connections
    app.run(host="0.0.0.0", port=port)
