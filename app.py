import os
import json
import torch
import numpy as np
import flask
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Text Classification Model
model_path = "Models/ClassificationModel"
classification_model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load NER Model
ner_model = tf.keras.models.load_model("Models/NER_Model/bilstm_ner1.keras")

# Load word2idx and idx2tag mappings
with open("Models/NER_Model/word2idx1.json", "r") as f:
    word2idx = json.load(f)

with open("Models/NER_Model/idx2tag1.json", "r") as f:
    idx2tag = json.load(f)

# Ensure indices are converted from string keys (JSON saves dict keys as strings)
word2idx = {k: int(v) for k, v in word2idx.items()}
idx2tag = {int(k): v for k, v in idx2tag.items()}

# Set maximum sequence length (same as used in training)
max_len = 12  # Make sure this matches what you used in training

# Text classification prediction function
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = classification_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    categories = ["Name", "Phone Number", "Amount", "Account Number"]
    return categories[prediction]

# NER prediction function
def predict_entities(sentence):
    words = sentence.split()
    
    # Handle empty input or words not in vocabulary
    if not words:
        return []
    
    seq = pad_sequences([[word2idx.get(w, 0) for w in words]], maxlen=max_len, padding='post')
    pred = ner_model.predict(seq)
    pred = np.argmax(pred, axis=-1)
    
    entities = []
    for i, word in enumerate(words):
        if i < len(pred[0]):  # Make sure we don't go out of bounds
            tag = idx2tag.get(pred[0][i], "O")
            if tag != "O":
                entities.append((word, tag))
    
    return entities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Save audio file temporarily
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    
    # In a real app, you would convert audio to text here
    # For now, we'll use the text sent along with the request for testing
    text = request.form.get('transcript', '')
    
    # Process the text
    category = classify_text(text)
    entities = predict_entities(text)
    
    # Clean up the temporary file
    os.remove(filepath)
    
    # Return the results
    return jsonify({
        'category': category,
        'entities': [{'word': word, 'tag': tag} for word, tag in entities],
        'text': text
    })

@app.route('/submit-form', methods=['POST'])
def submit_form():
    # Process form submission
    form_data = request.json
    
    # In a real app, you would save this to a database
    # For now, just return a success message
    return jsonify({'success': True, 'message': 'Form submitted successfully'})

if __name__ == '__main__':
    app.run(debug=True)
