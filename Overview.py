import streamlit as st
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import wordcloud

import math

from preprocessing import lda_process, Lemmatization_Tokenizer

from annotated_text import annotated_text, parameters


############# UI & Import Data ###############
# Default Number of topic
if "number_of_topic" not in st.session_state:
    st.session_state["number_of_topic"] = 7

# Default Abstract Number
if "abstract_num" not in st.session_state:
    st.session_state["abstract_num"] = 0

# Import from the previous process
lda_result = lda_process(number_topic = st.session_state.number_of_topic, csv_address= "data/ifpri_brief_df.csv")

df = lda_result.base_data
lda = lda_result.lda
component_df = lda_result.component_df
document_topic_df = lda_result.document_topic_df

# Create a selection section 
def multiselect_keyword(section, label):

    unique_keyword = set([keyword for list in lda_result.base_data[section] for keyword in list])

    options = st.multiselect(
        label = label,
        options = unique_keyword,
    )
    return options


# Side Bar
with st.sidebar:
    st.session_state["number_of_topic"] = st.select_slider(
                                            "Select the Number of Topic to Identify",
                                            options = range(1, 10),
                                            value = st.session_state["number_of_topic"])

    # Choose a keyword
    keyword = multiselect_keyword(section="Subject - keywords", label="Choose a Keyword")

    # Choose a region
    region = multiselect_keyword(section="Subject - country location", label="Choose a Region")

    # OR AND
    operator = st.radio(label = "Logical Relationship Among Filters", options = ["OR", "AND"], index = 0)




# Title
st.title("Topic Modeling with the Abstract of the IFPRI Policy Brief")

# Explanation
st.markdown('''
**What is the recipe of a policy brief?** This may be an odd sounding question to the traditional policymakers, but it may be an important question for someone (or a Generative AI :robot_face:) who wants to recreate or improve an existing high-quality policy brief.\n
\n Here, I collect the abstracts and the metadata of all English writing policy briefs in the [International Food Policy Research Institute's Publication Repository](https://ebrary.ifpri.org/digital/collection/p15738coll2/search/searchterm/brief/field/type/mode/all/conn/and/order/date/ad/desc). 
Then, I conducted a Latent Dirichlet Allocation (LDA) on the available abstracts to find out what topics and words consist a policy brief. Before conducting the analysis, I applied TF-IDF weighting to the abstract during the vectorization process. I also removed stopwords/numbers, applied lemmitization, and removed words with the term frequency lower than 15. 
\n
You can also compare the result of LDA analysis and the catorgy manually tagged by human IFPRI staff. Do you notice any pattern?
\n
*Is there such thing as a recipe of a policy brief?*  
\n
Please feel free to share your idea! 
You can check my code at [my GitHub Repository](https://github.com/minahkim-dspp/ifpri-abstract-topic-modeling).
''')


############# Compare with the Original Data ###############
# Header
st.header("Abstracts & Meta Data")

# Subsetted Dataframe
boolean_for_subsetting = np.full((df.shape[0],1), False, dtype = bool)

for keyword_column, options in [("Subject - keywords", keyword), ("Subject - country location", region)]:
    subset_boolean = np.array([any(option in sublist for option in options) for sublist in df[keyword_column]])
    if operator == "AND":
        boolean_for_subsetting = np.logical_and(boolean_for_subsetting, subset_boolean)
    else:
        boolean_for_subsetting = np.logical_or(boolean_for_subsetting, subset_boolean)

boolean_for_subsetting = boolean_for_subsetting[0, :]
subset_df = df.loc[boolean_for_subsetting]
st.dataframe(subset_df)



############# PCA(Principal Component Analysis) ###############

## 1. Standardize document distribution
standardization = StandardScaler()
standardized_document_topic_distribution = standardization.fit_transform(document_topic_df)
standardized_document_topic_distribution = pd.DataFrame.from_records(standardized_document_topic_distribution, columns= range(0, lda_result.lda.n_components))

## 2. Initiate PCA
pca = PCA(n_components = 2)
dimension_reduced_topics = pd.DataFrame.from_records(pca.fit_transform(standardized_document_topic_distribution), columns=["x", "y"])

# Append with the original dataset
dimension_reduced_topics = pd.concat([dimension_reduced_topics, document_topic_df], axis=1)

## 3. Graph
col1, col2 = st.columns([0.7, 0.3])

with col2:
    checkbox = np.empty(st.session_state["number_of_topic"], dtype = object)

    with st.form("checkbox_group"):

        for i in np.arange(0, st.session_state["number_of_topic"]):
            checkbox[i] = st.checkbox("Topic "+str(i), value = True)
        
        submitted = st.form_submit_button("Submit")


with col1:
    fig, ax = plt.subplots()
    tol_light_color ={0: "#77AADD", 1: "#EE8866", 2: "#EEDD88", 3: "#FFAABB", 4: "#99DDFF", 5: "#44BB99", 6: "#BBCC33", 7: "#AAAA00", 8:"#DDDDDD"}

    for i in np.arange(0, st.session_state["number_of_topic"]):
        if checkbox[i]:
            ax.scatter(x= dimension_reduced_topics.x, y= dimension_reduced_topics.y, c = tol_light_color[i], alpha = dimension_reduced_topics[i], label = i)

    ax.scatter(x= dimension_reduced_topics.iloc[st.session_state.abstract_num].x, 
                y= dimension_reduced_topics.iloc[st.session_state.abstract_num].y, 
                facecolors = "none", edgecolors='red', label = "Abstract-Selected")
    
    if "boolean_for_subsetting" in globals():
        subset_by_keyword= dimension_reduced_topics[boolean_for_subsetting]
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


############# Annotation ###############
st.header("Abstract")

#Choose an Abstract using title
title_abstract= st.selectbox(
                    label = "Choose the Title of the Abstract",
                    options = lda_result.base_data["Title"])

st.session_state.abstract_num = lda_result.base_data[lda_result.base_data.Title == title_abstract].index.values[0]
abstract = df.Abstract.iloc[st.session_state.abstract_num]

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

# Key
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Subject - author supplied keywords", "Subject - country location", "Subject - keywords", "IFPRI Descriptors", "Basic Information"])


if type(df["Subject - author supplied keywords"].iloc[st.session_state.abstract_num]) == str:
    with tab1:
        st.write(df["Subject - author supplied keywords"].iloc[st.session_state.abstract_num])

if type(df["Subject - country location"].iloc[st.session_state.abstract_num]) == list:
    with tab2:
        st.write("; ".join(df["Subject - country location"].iloc[st.session_state.abstract_num]))

if type(df["Subject - keywords"].iloc[st.session_state.abstract_num]) == list:
    with tab3:
         st.write("; ".join(df["Subject - keywords"].iloc[st.session_state.abstract_num]))

if type(df["IFPRI Descriptors"].iloc[st.session_state.abstract_num]) == str:
    with tab4:
         st.write(df["IFPRI Descriptors"].iloc[st.session_state.abstract_num])

with tab5:
    st.write("Title: "+ df["Title"].iloc[st.session_state.abstract_num])
    st.write("Author : " + df["Author"].iloc[st.session_state.abstract_num])
    st.write("Year : " + str(df["Year"].iloc[st.session_state.abstract_num]))


############# Distribution Bar Chart ###############
st.header("The Distribution of Topics of this Abstract")

# Start a plot
fig_1, ax_1 = plt.subplots()

# Extract the information about the document from document_topic_df
this_document_topic_df = document_topic_df[document_topic_df.index == st.session_state.abstract_num]

# Create a bar graph
tol_light_color_list =["#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00", "#DDDDDD"]

ax_1.bar(this_document_topic_df.columns, 
            this_document_topic_df.iloc[0],
            color= tol_light_color_list[0:st.session_state.number_of_topic])

# Set y label
ax_1.set_ylabel("The Distribution of the Topic")
# Set the limit of y axis so that it won't change across documents
ax_1.set_ylim([0,1])
# Set x label
ax_1.set_xlabel("Topic")

# Show the plot
st.pyplot(fig_1)   
