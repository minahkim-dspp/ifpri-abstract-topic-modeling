import streamlit as st
import pandas as pd
import numpy as np
import math

from pages.preprocessing import lda_process, Lemmatization_Tokenizer

import random

from annotated_text import annotated_text, parameters

import wordcloud
from matplotlib import pyplot as plt

### 1. Data to Visualize 

# Data, LDA object and the original Document Term Matrix
# Topics and their words 
# Documents and their topics
lda_result = lda_process(number_topic = 7, csv_address= "ifpri_brief_df.csv")

df = lda_result.base_data
lda = lda_result.lda
component_df = lda_result.component_df
document_topic_df = lda_result.document_topic_df

# Randomly choose the text that we will analyze
document_num = random.choice(range(0, df.shape[0]))
abstract = df.Abstract.iloc[document_num]

### 2. Building the backbone of the website

# Title
st.title("Topic Modeling with the Abstract of the IFPRI Policy Brief")

# Color Scheme
tol_light_color =["#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00", "#DDDDDD"]

# First Header
st.header("Abstract")

## Abstract
# Importing the lemmatization and tokenization process that the text went through
lemmatization_and_tokenizing = Lemmatization_Tokenizer()
# Spliting the abstract into a token level. This abstract should be kept as it is.
abstract_in_word = lemmatization_and_tokenizing.tokenizer(abstract)
# Create a "separate" list that has the tokenized/lemmatized/non-stopword words. 
words_that_count = [lemmatization_and_tokenizing(word) for word in abstract_in_word]

# Save the topic that the word represents by order of the word
topic_by_order_of_word = []

for word_in_dict in words_that_count:
    try:
        # If the word does not exist (i.e. excluded because it is a number or stopword), the topic should be "NoTopic"
        if word_in_dict == []:
            topic_by_order_of_word.append("NoTopic")
        elif word_in_dict != []:
            # extract the string form of the token
            tokenized_word = word_in_dict[0]
            # add the topic number of the topic that the word has the highest contribution
            topic_by_order_of_word.append(component_df[tokenized_word].sort_values(ascending = False).index[0])
    except:
        # Any other unidentifiable case will also fall under "NoTopic"
        topic_by_order_of_word.append("NoTopic")

# Create the list to use for annotated text using the two lists above
abstract_text_with_annotation = []

for i in range(0, len(topic_by_order_of_word)):
    # We keep only the original word if it has no topic.
    if topic_by_order_of_word[i] == "NoTopic":
        abstract_text_with_annotation.append(abstract_in_word[i]+" ")
    else:
    # If it has a topic assigned, we create a tuple with the (word, topic, color of the topic)
        topic_for_the_word = int(topic_by_order_of_word[i])
        abstract_text_with_annotation.append((abstract_in_word[i], str(topic_for_the_word), tol_light_color[topic_for_the_word]))

annotated_text(abstract_text_with_annotation)


# Second Header
st.header("What are the topics that we identified?")

## Word Cloud
fig = plt.figure()

for i in range(0, lda.n_components):
    # Calculate the number of row necessary
    row_n = math.ceil(lda.n_components/2)

    # Add a subplot
    ax = fig.add_subplot(row_n, 2, i+1)
    
    # The first 10 words that represent the data
    representation = component_df.iloc[i].sort_values(ascending= False).head(20)

    # Create a wordcloud
    wc = wordcloud.WordCloud(background_color = "white", 
                                color_func = lambda *args, **kwargs: tol_light_color[i]).generate_from_frequencies(representation)

    ax.imshow(wc)
    ax.axis('off')

st.pyplot(fig)

# Third Header
st.header("The Distribution of Topics of this Abstract")

# Start a plot
fig_1, ax_1 = plt.subplots()

# Extract the information about the document from document_topic_df
this_document_topic_df = document_topic_df[document_topic_df.index == document_num]

# Create a bar graph
ax_1.bar(this_document_topic_df.columns, 
            this_document_topic_df.iloc[0],
            color= tol_light_color[0:lda.n_components])

# Set y label
ax_1.set_ylabel("The Distribution of the Topic")
# Set the limit of y axis so that it won't change across documents
ax_1.set_ylim([0,1])
# Set x label
ax_1.set_xlabel("Topic")

# Show the plot
st.pyplot(fig_1)   




