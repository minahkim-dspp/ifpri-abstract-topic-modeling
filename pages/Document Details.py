import streamlit as st
import pandas as pd
import numpy as np
import math

from preprocessing import lda_process, Lemmatization_Tokenizer

from annotated_text import annotated_text, parameters

from matplotlib import pyplot as plt

############# UI ###############
with st.sidebar:

    if "number_of_topic" not in st.session_state:
        st.session_state["number_of_topic"] = 7

    st.session_state["number_of_topic"] = st.select_slider(
                                            "Select the Number of Topic to Identify",
                                            options = range(1, 10),
                                            value = st.session_state["number_of_topic"])



### 1. Data to Visualize 

# Data, LDA object and the original Document Term Matrix
# Topics and their words 
# Documents and their topics
lda_result = lda_process(number_topic = st.session_state.number_of_topic, csv_address= ".../data/ifpri_brief_df.csv")

df = lda_result.base_data
lda = lda_result.lda
component_df = lda_result.component_df
document_topic_df = lda_result.document_topic_df


### 2. Building the backbone of the website

# Title
#st.title("Topic Modeling with the Abstract of the IFPRI Policy Brief")

# Color Scheme
tol_light_color =["#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00", "#DDDDDD"]

# First Header
st.header("Abstract")

# Abstract Title

# Set the abstract number at 0 when not initialized
if "abstract_num" not in st.session_state:
    st.session_state.abstract_num = 0

#Choose an Abstract using title
title_abstract= st.selectbox(
                    label = "Choose the Title of the Abstract",
                    options = lda_result.base_data["Title"],
                    index = int(st.session_state.abstract_num))

st.session_state["abstract_num"] = lda_result.base_data[lda_result.base_data.Title == title_abstract].index.values[0]


document_num = st.session_state["abstract_num"]
abstract = df.Abstract.iloc[document_num]

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Subject - author supplied keywords", "Subject - country location", "Subject - keywords", "IFPRI Descriptors", "Basic Information"])


if type(df["Subject - author supplied keywords"].iloc[document_num]) == str:
    with tab1:
        st.write(df["Subject - author supplied keywords"].iloc[document_num])

if type(df["Subject - country location"].iloc[document_num]) == str:
    with tab2:
        st.write(df["Subject - country location"].iloc[document_num])

if type(df["Subject - keywords"].iloc[document_num]) == list:
    with tab3:
         st.write("; ".join(df["Subject - keywords"].iloc[document_num]))

if type(df["IFPRI Descriptors"].iloc[document_num]) == str:
    with tab4:
         st.write(df["IFPRI Descriptors"].iloc[document_num])

with tab5:
    st.write("Author : " + df["Author"].iloc[document_num])
    st.write("Year : " + str(df["Year"].iloc[document_num]))


# Forth Header
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


