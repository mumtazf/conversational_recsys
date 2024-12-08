import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load the files
with open("data/train_utt.txt", "r") as utt_file, open("data/train_ans.txt", "r") as ans_file:
    utterances = utt_file.readlines()
    intents_and_slots = ans_file.readlines()

# Function to clean and tokenize text
def clean_and_tokenize(text):
    # Standardize punctuation (remove special characters)
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-word characters except spaces
    return word_tokenize(text)  # Tokenize using NLTK

# Function to generate BIO labels
def generate_bio_labels(utterance, slots):
    tokens = clean_and_tokenize(utterance.strip())  # Clean and tokenize the utterance
    labels = ["O"] * len(tokens)  # Default all labels to "O"

    # Parse the slots
    slots = slots.strip().split("|")[1:]  # Skip the intent
    slot_map = {}
    for slot in slots:
        key, value = slot.split("=")
        value_tokens = clean_and_tokenize(value)  # Clean and tokenize the slot value
        slot_map[key] = value_tokens

    # Assign BIO tags
    for slot, value_tokens in slot_map.items():
        for i in range(len(tokens) - len(value_tokens) + 1):
            if tokens[i:i + len(value_tokens)] == value_tokens:
                labels[i] = f"B-{slot}"
                for j in range(1, len(value_tokens)):
                    labels[i + j] = f"I-{slot}"
                break

    return tokens, labels

# Process all utterances and slots
data = []
for utterance, slots in zip(utterances, intents_and_slots):
    tokens, labels = generate_bio_labels(utterance, slots)
    data.append({"tokens": tokens, "labels": labels})

# Convert to a DataFrame
df = pd.DataFrame(data)

# Save the processed data to a CSV file
df.to_csv("bert_dialogue_system/cleaned_labeled_dataset.csv", index=False)
print("Cleaned and labeled data saved to cleaned_labeled_dataset.csv")
