import io
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import json

# === Configuration ===
MODEL_PATH = 'best_model.keras'
IMG_SIZE = (128, 128)
PORT = 5000
HOST = '0.0.0.0'

# Load class names from JSON file
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

# Load the trained model once
model = load_model(MODEL_PATH)

# Flask app for receiving images
app = Flask(__name__)

def preprocess_image(image_bytes):
    # Decode, resize, normalize
    np_img = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'GET':
            return "Upload endpoint. Please send a POST request with an image.", 200
        elif request.method == 'POST':
            image_bytes = request.data
            img_tensor = preprocess_image(image_bytes)
            preds = model.predict(img_tensor)
            class_idx = np.argmax(preds, axis=1)[0]
            confidence = float(np.max(preds))
            # Get the class name based on class index
            class_name = class_names.get(class_idx, "Unknown")

            result = {'class_name': class_name, 'confidence': confidence}
            # Update GUI asynchronously
            gui_queue.put((image_bytes, result))
            return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask():
    app.run(host=HOST, port=PORT, threaded=True)

# GUI setup
class PredictionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plant ID Dashboard")
        self.geometry("600x500")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Image display
        self.img_label = ttk.Label(self)
        self.img_label.grid(row=0, column=0, pady=10)

        # Prediction text
        self.pred_var = tk.StringVar(value="Waiting for image...")
        self.pred_label = ttk.Label(self, textvariable=self.pred_var, font=("Helvetica", 16))
        self.pred_label.grid(row=1, column=0, pady=10)

        # Confidence bar
        self.conf_bar = ttk.Progressbar(self, orient='horizontal', length=300, mode='determinate')
        self.conf_bar.grid(row=2, column=0, pady=10)

        # Capture and predict button
        self.capture_button = ttk.Button(self, text="Capture and Predict", command=self.capture_and_predict)
        self.capture_button.grid(row=3, column=0, pady=10)

        # Poll queue
        self.after(100, self.process_queue)

    def process_queue(self):
        try:
            image_bytes, result = gui_queue.get_nowait()
        except:
            self.after(100, self.process_queue)
            return

        # Display image
        pil_img = Image.open(io.BytesIO(image_bytes)).resize((300,300))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.img_label.configure(image=tk_img)
        self.img_label.image = tk_img

        # Display prediction
        class_name = result['class_name']
        conf = result['confidence'] * 100
        self.pred_var.set(f"Class: {class_name}  ({conf:.1f}% )")
        self.conf_bar['value'] = conf

        self.after(100, self.process_queue)

    def capture_and_predict(self):
        # Simulate the capture and prediction process (triggered when button is pressed)
        response = requests.get('http://192.168.224.254:5000/upload')  # Trigger ESP32 to capture and send image
        print(response.text)

# === Main entry ===
if __name__ == '__main__':
    # Thread-safe queue for GUI updates
    import queue
    gui_queue = queue.Queue()

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Launch GUI (blocks)
    gui = PredictionGUI()
    gui.mainloop()
