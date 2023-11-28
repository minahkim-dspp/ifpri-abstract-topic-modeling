import streamlit as st
import pandas as pd

from preprocessing import lda_process

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt


## 0. Import from the previous process
lda_result = lda_process(number_topic = 7, csv_address="ifpri_brief_df.csv")
document_topic_df = lda_result.document_topic_df

## 1. Standardize document distribution
standardization = StandardScaler()
standardized_document_topic_distribution = standardization.fit_transform(document_topic_df)
standardized_document_topic_distribution = pd.DataFrame.from_records(standardized_document_topic_distribution, columns= range(0, lda_result.lda.n_components))

## 2. Initiate PCA
pca = PCA(n_components = 2)
dimension_reduced_topics = pd.DataFrame.from_records(pca.fit_transform(standardized_document_topic_distribution), columns=["x", "y"])

# Include the column for the most apparent topic
dimension_reduced_topics["topic"] = document_topic_df.apply(lambda row: row[row == row.max()].index.tolist(), axis = 1)
dimension_reduced_topics["topic"] = [element[0] for element in dimension_reduced_topics.topic]


# 3. Graph
fig, ax = plt.subplots()
tol_light_color ={0: "#77AADD", 1: "#EE8866", 2: "#EEDD88", 3: "#FFAABB", 4: "#99DDFF", 5: "#44BB99", 6: "#BBCC33", 7: "#AAAA00", 8:"#DDDDDD"}
scatter = ax.scatter(x= dimension_reduced_topics.x, y= dimension_reduced_topics.y, c = dimension_reduced_topics["topic"].map(tol_light_color))
ax.set_axis_off()
ax.legend(scatter.legend_elements())
st.pyplot(fig)