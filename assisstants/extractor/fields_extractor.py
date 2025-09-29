import sys

from annotated_types import doc
from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging

import re
import spacy

nlp = spacy.load("en_core_web_sm")

class ExtractFields:
    def extract(self, text):
        try:
            logging.info("Field Extraction Started")

            # Phone (Indian-style 10 digits, optional +91 or 0)
            phone_no = re.findall(r'(?:(?:\+91|0)?[\s\-]?)?[6-9]\d{9}', text)

            # Account numbers (usually 9 to 18 digits)
            account_no = re.findall(r'\b\d{9,18}\b', text)

            # Amounts (₹, Rs., or plain numbers with commas/decimals)
            amount = re.findall(r'(?:₹|Rs\.?|INR)?[\s]?[0-9,]+(?:\.\d{1,2})?', text)

            # Names
            doc = nlp(text)
            name = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            logging.info("Field Extraction Completed")
            return {
                "phones": phone_no,
                "accounts": account_no,
                "amounts": amount,
                "names": name
            }
        except Exception as e:
            raise AssisstantException(e, sys)