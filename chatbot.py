import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

#in terminal
#import
#inltk.download('punkt_tab')


# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load user inputs and responses from files
with open("script.txt", "r", encoding='utf-8') as script_file:
    user_inputs = [line.strip().lower() for line in script_file.readlines()]

with open("responses.txt", "r", encoding='utf-8') as response_file:
    chatbot_responses = [line.strip() for line in response_file.readlines()]

# Part-of-Speech mapping for WordNet
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Preprocess User Input
def preprocess_input(user_input):
    tokens = word_tokenize(user_input.lower())
    tagged_tokens = pos_tag(tokens)
    filtered_tokens = []

    for word, tag in tagged_tokens:
        if word not in stop_words and word.isalnum():
            pos = get_wordnet_pos(tag)
            if pos:
                filtered_tokens.append(lemmatizer.lemmatize(word, pos=pos))
            else:
                filtered_tokens.append(lemmatizer.lemmatize(word))
    return filtered_tokens

# Get synonyms using WordNet
def get_synonyms(word, pos):
    synonyms = set()
    for synset in wordnet.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return synonyms

# Feature extraction
def extract_features(user_input):
    tokens = preprocess_input(user_input)
    tagged = pos_tag(tokens)
    features = []
    for word, tag in tagged:
        pos = get_wordnet_pos(tag)
        if pos:
            features.extend(get_synonyms(word, pos))
    return set(features)

# Normalize user input for exact matching
def normalize_input(user_input):
    # Normalize common variations
    user_input = user_input.lower().strip()
    user_input = re.sub(r"\b(i am|i'm)\b", "im", user_input)  # Normalize "i am" and "i'm" to "im"
    user_input = re.sub(r"\b(you are|you're)\b", "youre", user_input)  # Normalize "you are" and "you're" to "youre"
    user_input = re.sub(r"\b(thankyou)\b", "thank you", user_input)  # Normalize "thankyou" to "thank you"
    return user_input

# Find the best response
def find_best_response(user_input):
    # Normalize the input for exact matching
    normalized_input = normalize_input(user_input)

    # First, check for exact matches
    if normalized_input in user_inputs:
        index = user_inputs.index(normalized_input)
        return chatbot_responses[index]

    # If no exact match, use feature-based matching
    user_features = extract_features(user_input.lower())
    for i, script in enumerate(user_inputs):
        script_features = extract_features(script)
        if user_features & script_features:  # Check for any overlapping words/synonyms
            return chatbot_responses[i]

    # Default response if no match is found
    return "I'm here to listen. Tell me more about what's on your mind."

# Main chatbot function
def chatbot():
    print("Hello, I'm your friendly chatbot! Feel free to talk to me about anything.")
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Take care and stay positive. Goodbye!")
            break
        response = find_best_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()