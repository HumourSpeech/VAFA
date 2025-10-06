import sys
from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging
from assisstants.utils.main_utils import convert_words_to_numbers

import contractions
import re

class TextProcessor:
    try:
        def process_text(self, text):
            logging.info("Text Processing Started")
            
            """
            Preprocess input text:
            1. Expand contractions (e.g., "I'm" → "I am")
            2. Convert to lowercase
            3. Remove punctuation & special characters
            4. Convert numbers (e.g., '5k' → '5000')
            5. Remove stopwords (optional)
            6. Handle multiple spaces
            """

            # Expand contractions (e.g., "I'm" → "I am")
            text = contractions.fix(text)

            # Convert to lowercase
            text = text.lower()

            # Remove special characters & punctuation (except numbers & words)
            text = re.sub(r"[^\w\s]", "", text)

            # Convert numbers in words and handle 5k, 10 lakh, etc.
            text = convert_words_to_numbers(text)

            # Remove extra spaces
            text = re.sub(r"\s+", " ", text).strip()

            logging.info(f"Text Processing Completed: {text}")
            return text
    except Exception as e:
        raise AssisstantException(e, sys)

    