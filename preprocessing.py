import pandas as pd
import re

import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

class DataProcessing:
    '''
    Import the CSV and conduct basic filtering and selection.
    This class exists to keep various version of the dataset in a single object
    '''
    def __init__(self, csv) -> None:
        self.raw_df = pd.read_csv(csv)
        self.df_of_interest = self.raw_df[['Title', 'Year', 'Author', 'Abstract', 'IFPRI Descriptors', 'Subject - author supplied keywords',
       'Subject - country location', 'Subject - keywords']]
        self.with_abstract = self.df_of_interest[self.df_of_interest.Abstract.isnull() == False]
        self.no_abstract = self.df_of_interest[self.df_of_interest.Abstract.isnull()]
    
class Lemmatization_Tokenizer(object):
    '''
    Integrate Lemmatizer from the NLTK package to the Scikit-Learn CountVectorizer/TF-IDF Vectorizer

    Return: List with lemmatize tokens
    '''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.countvect = CountVectorizer(stop_words='english')
        self.tokenizer = self.countvect.build_tokenizer()
        self.stop_word = self.countvect.get_stop_words()

        # Add stop words
        #my_additional_stop_words = ("the", "a")
        #self.stop_word = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

    def __call__(self, abstract):
        abstract_in_token = [word for word in self.tokenizer(abstract)]
        abstract_no_stopword = [word for word in abstract_in_token if word.casefold() not in self.stop_word]
        abstract_no_number = [word for word in abstract_no_stopword if bool(re.search(r"[0-9]+", word)) == False]

        return [self.wnl.lemmatize(token) for token in abstract_no_number]

def display_topics(model, feature_names, no_top_words):
    for topic_index,topic in enumerate(model.components_):
        feature_topic = []

        # sort it, and jump it every -1?
        topic_word_indexes = topic.argsort()[::-1]
        top_indexes = topic_word_indexes[:no_top_words]

        feature_topic.append([feature_names[i] for i in top_indexes])
        
        return feature_topic
         
def lda_process(number_topic = 6):

    ## 1. Read in the dataset & subset it
    df_base = DataProcessing("ifpri_brief_df.csv")

    ## 2. Lemmitization & Vectorization
    # Initiate the TF-IDF with lemmitization
    # The minimum term frequency was set to 15
    text_vector = TfidfVectorizer(stop_words= "english", min_df = 15, tokenizer=Lemmatization_Tokenizer(), token_pattern = None)

    ## 3. TF-IDF Document Term Matrix
    # Array
    text_vector_result = text_vector.fit_transform(df_base.with_abstract.Abstract).toarray()
    feature_names = text_vector.get_feature_names_out()

    # Dataframe
    index_word = {index: text for text, index in text_vector.vocabulary_.items()}
    index_word = [index_word[num] for num in range(0, len(index_word))]
    text_vector_result = pd.DataFrame(text_vector_result, columns = index_word)

    ## 4. Latent Dirichlet Allocation (LDA)
    lda = LatentDirichletAllocation(n_components = number_topic, random_state = 0)
    lda.fit(text_vector_result) 
    display_topics(lda, feature_names, 15)


    return df_base.with_abstract, lda, text_vector_result






