#import required libraries
import os                          #building file paths
import string                      #punctuation removal
from collections import Counter    #counts token frequencies

import nltk                        #NLP library
from nltk.tokenize import word_tokenize   #splits texts into tokens
from nltk.corpus import stopwords         #filters out meaningless words
from nltk.stem import PorterStemmer      #for stemming
from nltk.stem import WordNetLemmatizer  #for lemmatization
from nltk.util import ngrams  #for ngrams
import spacy                              #for named entity recognition

#Download Required NLTK Data
print("Downloading required NLTK data (only needed once)...")
nltk.download('punkt')          #tokenizer model
nltk.download('punkt_tab')      #tokenizer lookup table
nltk.download('stopwords')      #for identifying common stopwords
nltk.download('wordnet')        #database for lemmatization
nltk.download('averaged_perceptron_tagger_eng')  #part of speech tagger
print("NLTK data ready.\n")

#Loads the spaCy English Language Model
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.\n")
except OSError:
    print("spaCy model not found. Attempting to download it now...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model downloaded and loaded.\n")

#Define the File Paths
assignment_folder = os.path.dirname(os.path.abspath(__file__))

#List of the three text files to analyze
text_files = [os.path.join(assignment_folder, "Text1.txt"),
    os.path.join(assignment_folder, "Text2.txt"),
    os.path.join(assignment_folder, "Text3.txt"),
    os.path.join(assignment_folder, "Text4.txt"),]

#Initialize NLP Tools
# Create instances of the stemmer and lemmatizer.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#Load English stopwords
stop_words = set(stopwords.words('english'))


#Defines a function to Get lemmatizer part of speech tags, coverts fro NLTK tag to Word Net tag for more accuracy
def get_wordnet_pos(treebank_tag):
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ     #Adjective
    elif treebank_tag.startswith('V'):
        return wordnet.VERB    #Verb
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN    #Noun
    elif treebank_tag.startswith('R'):
        return wordnet.ADV     #Adverb
    else:
        return wordnet.NOUN    #Default to noun if tag is unrecognized

def get_top_ngrams(tokens, n, top_n=20):
    all_ngrams = list(ngrams(tokens, n))
    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top_n)

bigram_counts = {}
#Process Each Text File
#loops through each file and performs NLK process on each
for file_path in text_files:
    #Get just the filename
    file_name = os.path.basename(file_path)

    #Opens and reads the full text content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        print(f"\n[File loaded successfully — {len(raw_text)} characters]\n")
    except FileNotFoundError:
        print(f"\n  ERROR: File not found at:\n  {file_path}")
        print("  Please check that the file name and folder path are correct.\n")
        continue  #Skip to the next file if this one is missing
    #Tokenization
    # Splits the raw text into individual word tokens.
    print("Tokenization:")
    tokens = word_tokenize(raw_text)
    print(f"Total tokens (including punctuation): {len(tokens)}")
    #Clean the Tokens
    # Remove punctuation and stopwords, then convert everything to lowercase.
    clean_tokens = [
        token.lower()                         #Lowercase for consistency
        for token in tokens
        if token.lower() not in stop_words    #Remove stopwords
        and token not in string.punctuation   #Remove punctuation marks
        and token.isalpha()                   #Keep only letters (no numbers/symbols)
    ]
    print(f"Clean tokens (after removing stopwords & punctuation): {len(clean_tokens)}\n")
    #Stemming
    #Reduces each word to its root/stem by stripping suffixes.
    print("Stemming:")
    stemmed_tokens = [stemmer.stem(token) for token in clean_tokens]

    #Lemmatization
    #Reduces each word to its true dictionary base form
    print("Lemmatization:")
    #Get part-of-speech tags for each clean token
    pos_tags = nltk.pos_tag(clean_tokens)

    #lemmatize each word using its POS tag for higher accuracy
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag))
        for token, pos_tag in pos_tags]

    #count frequency of each lemmatized token and get the top 20
    lemma_freq = Counter(lemmatized_tokens)
    top_20_lemmas = lemma_freq.most_common(20)

    print("Top 20 most common tokens:")
    for rank, (token, count) in enumerate(top_20_lemmas, start=1):
        print(f"  {rank:>2}. {token:<20} — {count} occurrence(s)")
    print()
    #Named Entity Recognition (NER) with spaCy, looks at original text, not cleaned tokens
    print("Named entity recognition:")
    doc = nlp(raw_text)
    #Extract all named entities found in the document
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    total_entities = len(entities)
    print(f"Total named entities found: {total_entities}")

    if total_entities > 0:
        #Group and count entities by their type label
        entity_type_counts = Counter(label for _, label in entities)
        print("\nEntity by type:")
        for entity_type, count in sorted(entity_type_counts.items()):
            print(f"  {entity_type:<15} — {count} instance(s)")
        # Show up to the first 15 individual entities as examples
        print("\nExample entities (first 15):")
        seen = set()
        shown = 0
        for text, label in entities:
            if text not in seen and shown < 15:
                print(f"  [{label}] {text}")
                seen.add(text)
                shown += 1
    else:
        print("  No named entities were detected in this text.")

    print()

#What texts are about, based on most common tokens and entities:
#A romeo and juliet story set in verona. Romeo and Juliet were destined to fall in love by ate.
#There was a tragic despair that left them eternally damaged.

#Part 2
    #N-Gram Analysis, two word phrases and 3 word phrases
    print("N gram analysis:")
    #Build a natural token list (lowercase, letters only, stopwords kept), with stopwords
    natural_tokens = [
        token.lower()
        for token in tokens
        if token.isalpha()]  #Letters only

    #finds 2 word sequences
    # Collect natural bigrams (with stopwords) for each file
    natural_tokens = [
        token.lower()
        for token in tokens
        if token.isalpha()
    ]
    bigram_counts[file_name] = Counter(ngrams(natural_tokens, 2))

# Finds which 2-word phrases (with stopwords) appear in Text4
# and compares how often they appear in the other three texts.
print("2 word phrases comparison:")

# Get the top 20 bigrams from Text4
text4_top = bigram_counts.get("Text4.txt", Counter()).most_common(20)

if not text4_top:
    print("Text4.txt was not loaded or had no bigrams.")
else:
    print(f"\n{'Phrase':<30} {'Text1':>8} {'Text2':>8} {'Text3':>8} {'Text4':>8}")
    for gram, text4_count in text4_top:
        phrase = " ".join(gram)
        t1 = bigram_counts.get("Text1.txt", Counter())[gram]
        t2 = bigram_counts.get("Text2.txt", Counter())[gram]
        t3 = bigram_counts.get("Text3.txt", Counter())[gram]
        print(f"  {phrase:<28} {t1:>8} {t2:>8} {t3:>8} {text4_count:>8}")

#based on the comparison between the 2 word phrases in the 4 texts, the same author wrote the first three texts, but not the fourth.
# Text4 has vastly different number of occurrences for the bigrams than do the first 3 texts.