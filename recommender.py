import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# Download stopwords once
nltk.download("stopwords")

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

# ---- Content-Based Filtering ----
def content_based_recommendations(title, movies, top_n=10):
    movies['genres'] = movies['genres'].fillna('')
    tfidf = TfidfVectorizer(stop_words=stopwords.words("english"))
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    
    if title not in indices:
        return []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

# ---- Collaborative Filtering ----
def collaborative_recommendations(user_id, ratings, movies, top_n=10):
    user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating')
    user_movie_ratings = user_movie_ratings.fillna(0)
    
    # Compute similarity
    similarity = cosine_similarity(user_movie_ratings)
    
    user_index = user_id - 1  # userId starts from 1
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_users = [i[0] for i in sim_scores[1:6]]
    similar_users_ratings = user_movie_ratings.iloc[top_users].mean().sort_values(ascending=False)
    
    recommended_movie_ids = similar_users_ratings.head(top_n).index
    return movies[movies['movieId'].isin(recommended_movie_ids)][['title', 'genres']]
