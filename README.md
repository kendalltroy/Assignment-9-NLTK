# Assignment-9-NLTK

# Purpose
To perform Natural Lnaguage Processing (NLP) on four text files
Uses tokenization, stemming, and lemmantization to identify the 20 most common tokens per text
Performs n-gram analysis to determine common 2 word phrases and compare the number of occurrences between texts
Can be used to determine what texts are written by the same author

# Functions
- get_wordnet_pos(treebank_tag): converts NLTK part of speech tags to WordNet format for higher accuracy and better lammetization
- get_top_ngrams(tokens, n, top_n=20): generates n grams from the token list and returns the most frequent ones

# Limitations
Only analyzes up to 15 entities per file 
Two-word phrases may be sparse on texts 1,2, and 3 compared to 4
spaCy's named entity recognition may missclassify text
Stemming can produce non-real words
