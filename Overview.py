import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import lda_process

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import wordcloud

import math

############# UI & Import Data ###############
# Side Bar
with st.sidebar:

    if "number_of_topic" not in st.session_state:
        st.session_state["number_of_topic"] = 7

    st.session_state["number_of_topic"] = st.select_slider(
                                            "Select the Number of Topic to Identify",
                                            options = range(1, 10),
                                            value = st.session_state["number_of_topic"])

# Title
st.title("Topic Modeling with the Abstract of the IFPRI Policy Brief")

if "abstract_num" not in st.session_state:
    st.session_state.abstract_num = 0

# Import from the previous process
lda_result = lda_process(number_topic = st.session_state.number_of_topic, csv_address="ifpri_brief_df.csv")
document_topic_df = lda_result.document_topic_df

############# PCA(Principal Component Analysis) ###############

## 1. Standardize document distribution
standardization = StandardScaler()
standardized_document_topic_distribution = standardization.fit_transform(document_topic_df)
standardized_document_topic_distribution = pd.DataFrame.from_records(standardized_document_topic_distribution, columns= range(0, lda_result.lda.n_components))

## 2. Initiate PCA
pca = PCA(n_components = 2)
dimension_reduced_topics = pd.DataFrame.from_records(pca.fit_transform(standardized_document_topic_distribution), columns=["x", "y"])

# Include the column for the most apparent topic
#dimension_reduced_topics["topic"] = document_topic_df.apply(lambda row: row[row == row.max()].index.tolist(), axis = 1)
#dimension_reduced_topics["topic"] = [element[0] for element in dimension_reduced_topics.topic]

# Append with the original dataset
dimension_reduced_topics = pd.concat([dimension_reduced_topics, document_topic_df], axis=1)

## 3. Graph
col1, col2 = st.columns([0.7, 0.3])

with col2:
    checkbox = np.empty(st.session_state["number_of_topic"], dtype = object)

    for i in np.arange(0, st.session_state["number_of_topic"]):
        checkbox[i] = st.checkbox("Topic "+str(i), value = False)

with col1:
    fig, ax = plt.subplots()
    tol_light_color ={0: "#77AADD", 1: "#EE8866", 2: "#EEDD88", 3: "#FFAABB", 4: "#99DDFF", 5: "#44BB99", 6: "#BBCC33", 7: "#AAAA00", 8:"#DDDDDD"}

    for i in np.arange(0, st.session_state["number_of_topic"]):
        if checkbox[i]:
            ax.scatter(x= dimension_reduced_topics.x, y= dimension_reduced_topics.y, c = tol_light_color[i], alpha = dimension_reduced_topics[i], label = i)


    ax.scatter(x= dimension_reduced_topics.iloc[st.session_state["abstract_num"]].x, 
                y= dimension_reduced_topics.iloc[st.session_state["abstract_num"]].y, 
                facecolors = "none", edgecolors='red', label = "Abstract-Selected")
    
    if "options" in st.session_state:
        subset_by_keyword= dimension_reduced_topics[[any(option in sublist for option in st.session_state.options) for sublist in lda_result.base_data["Subject - keywords"]]]
        ax.scatter(x= subset_by_keyword.x, 
                y= subset_by_keyword.y, 
                facecolors = "none", edgecolors='blue', label = "Keyword-Selected")

    handles = [Patch(color=tol_light_color[topic]) for topic in range(0, st.session_state["number_of_topic"])]
    labels = ['Topic '+str(topic) for topic in range(0, st.session_state["number_of_topic"])]
    ax.legend(handles, labels)

    st.pyplot(fig)

############# Topic Word Cloud ###############
## Word Cloud
fig = plt.figure()

for i in range(0, lda_result.lda.n_components):
    # Calculate the number of row necessary
    row_n = math.ceil(lda_result.lda.n_components/2)

    # Add a subplot
    ax = fig.add_subplot(row_n, 2, i+1)
    
    # The first 10 words that represent the data
    representation = lda_result.component_df.iloc[i].sort_values(ascending= False).head(20)

    # Create a wordcloud
    wc = wordcloud.WordCloud(background_color = "white", 
                                color_func = lambda *args, **kwargs: tol_light_color[i]).generate_from_frequencies(representation)

    ax.imshow(wc)
    ax.axis('off')

st.pyplot(fig)

############# Compare with the Original Data ###############
# Header
st.header("Identify Abstracts using Meta Data")
#Choose an Abstract using title
title_abstract= st.selectbox(
                    label = "Choose the Title of the Abstract",
                    options = lda_result.base_data["Title"],
                    index = int(st.session_state.abstract_num))

st.session_state["abstract_num"] = lda_result.base_data[lda_result.base_data.Title == title_abstract].index.values[0]
st.write(lda_result.base_data[lda_result.base_data.Title == title_abstract])

# Choose a keyword

unique_keyword = set([keyword for list in lda_result.base_data["Subject - keywords"] for keyword in list])

if "options" not in st.session_state:
    st.session_state.options = []

st.session_state["options"] = st.multiselect(
    label = "Choose a Keyword",
    options = unique_keyword,
    default = st.session_state.options
)

if st.session_state["options"]:
    st.dataframe(
        lda_result.base_data[[any(option in sublist for option in st.session_state.options) for sublist in lda_result.base_data["Subject - keywords"]]]
    )

