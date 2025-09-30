import sys
from assisstants.exception.exception import AssisstantException

import re
from word2number import w2n

def convert_words_to_numbers(text):
    try:
        # Number word mapping for conversions (expandable)
        number_mapping = {
            "k": "000",   # 5k -> 5000
            "m": "000000",  # 2m -> 2000000
            "b": "000000000",  # 3b -> 3000000000
            "lakh": "00000",  # 10 lakh -> 1000000
            "crore": "0000000",  # 2 crore -> 20000000
            "million": "000000",  # half million -> 500000
            "billion": "000000000"  # 3 billion -> 3000000000
        }
        
        """
        Convert spoken numbers (e.g., 'five thousand' → '5000') & handle 5k, 10 lakh, etc.
        """
        words = text.split()
        processed_words = []
        temp_phrase = ""

        for word in words:
            # Handle abbreviations like "5k" → "5000"
            for key, value in number_mapping.items():
                if word.endswith(key):
                    num_part = re.sub(r"\D", "", word)  # Extract numeric part
                    if num_part:
                        processed_words.append(num_part + value)
                        temp_phrase = ""  # Clear the phrase
                    break
            else:
                # Accumulate words to form a numeric phrase
                temp_phrase += f" {word}"
                try:
                    # Attempt to convert accumulated words to a number
                    num_value = w2n.word_to_num(temp_phrase.strip())
                    processed_words.append(str(num_value))
                    temp_phrase = ""  # Clear after conversion
                except ValueError:
                    continue

        # Append any remaining phrase
        if temp_phrase.strip():
            processed_words.extend(temp_phrase.strip().split())

        return " ".join(processed_words)
    except Exception as e:
        raise AssisstantException(e, sys)

