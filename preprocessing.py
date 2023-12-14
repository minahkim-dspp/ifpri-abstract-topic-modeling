import pandas as pd
import numpy as np
import re

import os

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

        self.with_abstract = self.converting_category(self.with_abstract)

    def converting_category(self, data):
        # Organize the keywords section
        data.loc["Subject - keywords"] = data["Subject - keywords"].str.lower()
        data.loc["Subject - keywords"] = [re.sub(r"\(|\)", "", text) if type(text) == str else "" for text in data["Subject - keywords"]]
        data.loc["Subject - keywords"] = [re.split(r"\n", text) for text in data["Subject - keywords"]]

        # Organize the Country Location Section
        data.loc["Subject - country location"] = [re.split(r"\n", text) if type(text) == str else [] for text in data["Subject - country location"]]

        return data
        
    
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

class LDAResults:
    def __init__(self, base_data, lda, dtm) -> None:
        self.base_data = base_data
        self.lda = lda
        self.dtm = dtm
        self.component_df = self.component_array()
        self.document_topic_df = self.document_topic_df()

    
    def component_array(self):
        # Normalized Component Array
        component_array = self.lda.components_/self.lda.components_.sum(axis =1)[:, np.newaxis]
        # In dataframe
        component_df = pd.DataFrame(component_array, columns = self.lda.feature_names_in_)
        return component_df

    def document_topic_df(self):
        # Documents and their topics
        document_topic_df = pd.DataFrame(self.lda.transform(self.dtm))    
        return document_topic_df



def lda_process(number_topic = 6, csv_address = "ifpri-abstract-topic-modeling/data/ifpri_brief_df.csv"):

    ## 1. Read in the dataset & subset it
    df_base = DataProcessing(csv_address)

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

    # 5. Derive the Implication through the results
    lda_result = LDAResults(df_base.with_abstract, lda, text_vector_result)

    return lda_result






