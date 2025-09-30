from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging
import sys

from assisstants.loader.model_loader import ModelLoader
from assisstants.processor.text_processor import TextProcessor
from assisstants.Classifier.text_classifier import TextClassifier
from assisstants.extractor.fields_extractor import ExtractFields


if __name__ == "__main__":
    try:
        text = input("Input Text: ")
        processed_text = TextProcessor().process_text(text)

        # @st.cache_resource  ## Uncomment while deploying on streamlit cloud(Caching for fast loading of model and tokenizer)
        def init():
            return ModelLoader.get_model(), ModelLoader.get_tokenizer()

        model, tokenizer = init()
        classifier = TextClassifier()
        label = classifier.classify(processed_text)

        extractor = ExtractFields()

        Name, Phone, Amount, Account = extractor.extract(processed_text)

        print(f"Name: {Name}, Phone: {Phone}, Amount: {Amount}, Account: {Account}, Label: {label}")

    except Exception as e:
        raise AssisstantException(e, sys)