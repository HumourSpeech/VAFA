import sys
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging
from assisstants.constants import MODEL_PATH

class ModelLoader:
    try:
        _model = None
        _tokenizer = None
        _device = None

        @classmethod
        def _init_device(cls):
            if cls._device is None:
                cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return cls._device

        @classmethod
        def get_tokenizer(cls):
            if cls._tokenizer is None:
                try:
                    logging.info("Loading tokenizer from %s", MODEL_PATH)
                    cls._tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
                except Exception as e:
                    raise AssisstantException(e, sys)
            return cls._tokenizer

        @classmethod
        def get_model(cls):
            if cls._model is None:
                try:
                    device = cls._init_device()
                    logging.info("Loading model from %s on device %s", MODEL_PATH, device)
                    cls._model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
                    cls._model.to(device)
                    cls._model.eval()  # important for inference (dropout off)
                except Exception as e:
                    raise AssisstantException(e, sys)
            return cls._model
    except Exception as e:
        raise AssisstantException(e, sys)