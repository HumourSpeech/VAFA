import sys

import torch
from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging

from assisstants.constants import MODEL_PATH
from assisstants.loader.model_loader import ModelLoader

# from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
# model_path = MODEL_PATH
# # Load the trained model
# model = DistilBertForSequenceClassification.from_pretrained(model_path)
# # Load the tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained(model_path)

class TextClassifier:
    def classify(self, text):
        try:
            logging.info("Text Classification Started")

            model = ModelLoader.get_model()
            tokenizer = ModelLoader.get_tokenizer()
            device = ModelLoader._init_device()

            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

            # move tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            categories = ["Name", "Phone Number", "Amount", "Account Number"]

            logging.info(f"Text Classification Completed: {categories[prediction]}")
            return categories[prediction]
        except Exception as e:
            raise AssisstantException(e, sys)