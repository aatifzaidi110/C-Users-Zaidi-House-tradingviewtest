#===setup.py===
import nltk
import ssl

# This is a workaround for a common SSL certificate issue with NLTK downloads.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Explicitly download the 'vader_lexicon' data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("NLTK's vader_lexicon is already downloaded.")
except LookupError:
    print("Downloading NLTK's vader_lexicon...")
    nltk.download('vader_lexicon')
    print("Download complete.")