import os
import io
import base64
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort

# --------------------------------------------------
# Constants and model loading
# --------------------------------------------------
CHARSET = '0123456789abcdefghijklmnopqrstuvwxyz'
IDX2CHAR = {i: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET)
NUM_POS = 5
EXPECTED_SIZE = (224, 224)
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'holako bag.onnx')

# Initialize ONNX runtime session
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    # Resize, convert to RGB, normalize
    img = img.resize(EXPECTED_SIZE).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    # Shape: HWC -> CHW, add batch dim
    x = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return x


def predict_captcha(x: np.ndarray) -> str:
    # Run ONNX model
    outputs = session.run(None, {'input': x})[0]
    # Reshape and take argmax
    outs = outputs.reshape(1, NUM_POS, NUM_CLASSES)
    idxs = np.argmax(outs, axis=2)[0]
    return ''.join(IDX2CHAR[i] for i in idxs)

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts multipart/form-data with file field 'image'
    OR JSON with base64 string in field 'image'.
    Returns JSON: {'result': '<predicted_text>'}
    """
    # Check form-data file
    if 'image' in request.files:
        img_file = request.files['image']
        img = Image.open(img_file.stream)
    else:
        # Expect JSON base64
        data = request.get_json(silent=True) or {}
        b64 = data.get('image', '')
        # Remove header if exists
        if ',' in b64:
            b64 = b64.split(',')[1]
        try:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
        except Exception:
            return jsonify({'error': 'Invalid image data'}), 400

    # Preprocess and predict
    x = preprocess_image(img)
    result = predict_captcha(x)
    return jsonify({'result': result})

if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
