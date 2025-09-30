import sys
from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging

class ExtractFields:
    def extract(self, text):
        try:
            logging.info("Field Extraction Started")
            import re

            # Phone (Indian-style 10 digits, optional +91 or 0)
            phone_no = re.findall(r'(?:(?:\+91|0)?[\s\-]?)?[6-9]\d{9}', text)

            # Account numbers (usually 11 to 18 digits)
            account_no = re.findall(r'\b\d{11,18}\b', text)

            # Amounts (₹, Rs., or plain numbers with commas/decimals)
            amount = re.findall(r'(?:₹|Rs\.?|INR)?[\s]?[0-9,]+(?:\.\d{1,2})?', text)

            # Names
            # Need to fine tune this NER model with specific Bank database for more better accuracy
            import spacy
            
            nlp = spacy.load("en_core_web_sm")

            doc = nlp(text)
            name = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

            logging.info("Field Extraction Completed")

            return {
                name[0] if name else None,
                phone_no[0] if phone_no else None,
                amount [0] if amount else None,
                account_no[0] if account_no else None,
            }
        except Exception as e:
            raise AssisstantException(e, sys)