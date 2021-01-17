# This was taken from https://github.com/rantsandruse/lstm_attention_tf/blob/master/lib/read_article.py
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk ,re
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')


def data_to_reviews( data, column, toLower = True, remove_nonletters = True, remove_html=True, remove_stopwords = False, add_stemmer=False):
    reviews = []
    for review in data[column]:
        reviews.append( review_to_wordlist( review, remove_stopwords = remove_stopwords, remove_nonletters = remove_nonletters, toLower = toLower, remove_html = remove_html, add_stemmer = add_stemmer))

    return reviews

def review_to_wordlist( review_text, remove_stopwords=False, remove_nonletters = True, toLower = True, remove_html = True, add_stemmer=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    # But if we have clean formatted
    # There is no need in doing so.

    if remove_html:
        review_text = BeautifulSoup(review_text, "lxml").get_text()
    #
    # 2. Remove non-letters
    # For tweets, we do not wish to remove non-letters.
    if remove_nonletters:
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    if(toLower):
        words = review_text.lower().split()
    else:
        words = review_text.split()
    #
    # 4. Remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # You can also create
        # stops = FINAL_STOPWORDS
        words = [w for w in words if not w in stops]

    # 5. Lemmetization - this is slow.
    if add_stemmer:
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]

    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer = nltk.data.load('tokenizers/punkt/english.pickle'), remove_stopwords = False, remove_nonletters = True, toLower=True):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    # Split things into a list of sentences
    #tokenizer = RegexpTokenizer(r'(\w|\')+')

    #use this for regular reviews
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    #Use this from twitter
    #tokenizer = TweetTokenizer()

    raw_sentences = tokenizer.tokenize(review)

    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # do not mark negation for now.
            # Otherwise, call review_to_wordlist to get a list of words
            #sentences.append( review_to_wordlist( mark_negation(raw_sentence), \
            #  remove_stopwords, remove_nonletters, toLower ) )
            sentences.append( review_to_wordlist( raw_sentence, \
            remove_stopwords = remove_stopwords, remove_nonletters = remove_nonletters, toLower = toLower ) )

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences