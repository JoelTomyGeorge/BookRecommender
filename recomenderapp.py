# prompt: create a streamlit app to do all this

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
#import seaborn as sns

# Load data (make sure to upload movies.csv and ratings.csv to your Streamlit app directory)
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# defining functions for recomending

def get_movie_recommendations(movie_name):
    if movie_name in pivot_table.columns:
        idx = pivot_table.columns.get_loc(movie_name)
        distances, indices = knn.kneighbors(pivot_table.iloc[:, idx].values.reshape(1, -1))
        recommendations = [pivot_table.columns[i] for i in indices[0][1:]]
        return recommendations
    else:
        return f"Movie '{movie_name}' not found in the dataset."

def get_content_based_recommendations(movie_name):
    if movie_name in movies['title'].values:
        idx = movies[movies['title'] == movie_name].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return [movies['title'].iloc[i[0]] for i in sim_scores[1:6]]
    else:
        return f"Movie '{movie_name}' not found in the dataset."

# Preprocessing (same as before)
movies.drop_duplicates(subset='movieId', inplace=True)
ratings.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
data = pd.merge(ratings, movies, on='movieId')
pivot_table = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(pivot_table.T)


movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit app
st.title("Movie Recommendation App")

# Movie selection
movie_name = st.selectbox("Select a movie:", movies['title'].unique())


# Recommendation type
recommendation_type = st.radio("Choose Recommendation Type:", ('Collaborative Filtering', 'Content-Based Filtering'))


if st.button('Get Recommendations'):
    if recommendation_type == 'Collaborative Filtering':
        recommendations = get_movie_recommendations(movie_name)
        if isinstance(recommendations, str): # Handle movie not found case
           st.write(recommendations)
        else:
           st.write("<div style='font-size: 24px;'>Collaborative Filtering Recommendations:</div>", unsafe_allow_html=True)
           for movie in recommendations:
              st.write(movie)

    elif recommendation_type == 'Content-Based Filtering':
        recommendations = get_content_based_recommendations(movie_name)
        if isinstance(recommendations, str): # Handle movie not found
            st.write(recommendations)
        else:
           st.write("<div style='font-size: 24px;'>Content-Based Filtering Recommendations:</div>", unsafe_allow_html=True)
           for movie in recommendations:
               st.write(movie)

# # Data Visualization (example)
# st.subheader("Data Visualization")

# # Ratings distribution
# fig, ax = plt.subplots()
# sns.histplot(data['rating'], kde=True, ax=ax)
# st.pyplot(fig)


