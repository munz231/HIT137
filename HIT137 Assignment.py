#Question 1

import os
import pandas as pd
import spacy
from spacy.lang.en import English
from transformers import AutoTokenizer
from collections import Counter

# Task 1: Extract text from CSV files and store in a single .txt file
def extract_text_from_csv(folder_path, output_file):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            if 'text' in df.columns:
                texts.extend(df['text'].tolist())

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))

# Task 2: Install libraries
# Run the following commands in your terminal or command prompt:
# pip install spacy
# pip install scispacy
# pip install transformers

# Task 3.1: Count occurrences of words and store in CSV
def count_and_store_top_words(input_file, output_csv):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    nlp = English()
    tokens = [token.text for token in nlp(text)]

    word_counts = Counter(tokens)
    top_30_words = word_counts.most_common(30)

    df = pd.DataFrame(top_30_words, columns=['Word', 'Count'])
    df.to_csv(output_csv, index=False)

# Task 3.2: Count unique tokens using AutoTokenizer
def count_and_store_top_tokens(input_file, output_csv):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

    token_counts = Counter(tokens)
    top_30_tokens = token_counts.most_common(30)

    df = pd.DataFrame(top_30_tokens, columns=['Token', 'Count'])
    df.to_csv(output_csv, index=False)

# Task 4: Named-Entity Recognition
def ner_comparison(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Load SpaCy models
    spacy_nlp = spacy.load('en_core_sci_sm')
    spacy_entities = spacy_nlp(text).ents

    # Load BioBERT model (you need to install it first)
    # Example: pip install transformers sentencepiece torch
    bio_model = AutoModelForTokenClassification.from_pretrained("monologg/biobert_v1.1_pubmed_ner")
    bio_tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed_ner")
    bio_input_ids = bio_tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        bio_outputs = bio_model(bio_input_ids)
    
    bio_predictions = torch.argmax(bio_outputs.logits, dim=2)
    bio_entities = [bio_tokenizer.decode(pred) for pred in bio_predictions[0]]

    # Compare entities
    common_entities = set(spacy_entities) & set(bio_entities)
    difference_entities = set(spacy_entities) ^ set(bio_entities)

    # Print results or further analysis as needed
    print("Total entities detected by SpaCy:", len(spacy_entities))
    print("Total entities detected by BioBERT:", len(bio_entities))
    print("Common entities:", common_entities)
    print("Entities unique to SpaCy:", set(spacy_entities) - common_entities)
    print("Entities unique to BioBERT:", set(bio_entities) - common_entities)

# Run the tasks
folder_path = 'path/to/your/csv/folder'
output_txt = 'output/text_file.txt'
output_csv_word_count = 'output/top_30_words.csv'
output_csv_token_count = 'output/top_30_tokens.csv'

# Task 1
extract_text_from_csv(folder_path, output_txt)

# Task 3.1
count_and_store_top_words(output_txt, output_csv_word_count)

# Task 3.2
count_and_store_top_tokens(output_txt, output_csv_token_count)

# Task 4
# Make sure to install transformers and torch before running this task
ner_comparison(output_txt)


#Question 2


#Chapter 1: The Gatekeeper

from PIL import Image

# Load the original image
original_image = Image.open('Chapter1.png')
pixels = original_image.load()

# The provided algorithm to generate a number
import time
current_time = int(time.time())
generated_number = (current_time * 100) + 50
if generated_number % 2 == 0:
    generated_number += 10

# Modify the pixels in the image
for i in range(original_image.width):
    for j in range(original_image.height):
        r, g, b = pixels[i, j]
        pixels[i, j] = (r + generated_number, g + generated_number, b + generated_number)

# Save the new image
original_image.save('chapter1out.png')

# Calculate the sum of red pixel values
red_pixel_sum = sum([pixels[i, j][0] for i in range(original_image.width) for j in range(original_image.height)])

# Output the sum for the next chapter
print(red_pixel_sum)


#Chapter 2: The Chamber of Strings

s = "5610873770745785410aAwwsktraYmnssfsqp"

# Separate the string into number and letter substrings
number_substring = ''.join(c for c in s if c.isdigit())
letter_substring = ''.join(c for c in s if c.isalpha())

# Convert even numbers in the number substring to ASCII code Decimal values
even_numbers_ascii = [ord(c) for c in number_substring if int(c) % 2 == 0]

# Convert upper-case letters in the letter substring to ASCII code Decimal values
uppercase_letters_ascii = [ord(c) for c in letter_substring if c.isupper()]

# Output the results
print(even_numbers_ascii)
print(uppercase_letters_ascii)


#crypto

def decipher_cryptogram(cryptogram, shift):
    result = ''
    for char in cryptogram:
        if char.isalpha():
            ascii_offset = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - ascii_offset - shift) % 26 + ascii_offset)
        else:
            result += char
    return result

# Example usage
cryptogram = "URMOGAG QRFREIR 7 R NG 21 CRFG ZAEVYI A ZBAFBR"
for shift in range(26):
    deciphered_text = decipher_cryptogram(cryptogram, shift)
    print(f"Shift {shift}: {deciphered_text}")














