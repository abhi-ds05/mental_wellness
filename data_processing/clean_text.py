import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize once
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text, verbose=False):
    """
    Clean and preprocess journal entry or Reddit comment text.

    Args:
        text (str): Raw input string.
        verbose (bool): If True, prints intermediate steps.

    Returns:
        str: Cleaned and lemmatized text string.
    """
    if not isinstance(text, str):
        return ""

    if verbose:
        print(f"Original text: {text}")

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Rejoin
    cleaned_text = " ".join(tokens)

    if verbose:
        print(f"Cleaned text: {cleaned_text}")

    return cleaned_text


if __name__ == "__main__":
    # Safe NLTK downloads only when run as main script
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    sample_text = "I'm feeling really anxious today! Check out https://example.com :) #stress @friend"
    cleaned = clean_text(sample_text, verbose=True)
