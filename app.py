import streamlit as st
from recommender import load_data, content_based_recommendations, collaborative_recommendations

st.set_page_config(page_title="🎬 ReelMatch", layout="wide")

st.title("🎬 ReelMatch: Hybrid Movie Recommendation System")
st.write("Get personalized movie suggestions using **content-based** and **collaborative filtering**.")

# Load Data
movies, ratings = load_data()

tab1, tab2 = st.tabs(["🔎 Content-Based", "🤝 Collaborative"])

with tab1:
    st.subheader("Content-Based Filtering")
    movie_title = st.selectbox("Choose a movie:", movies['title'].values)
    if st.button("Recommend Similar Movies"):
        recommendations = content_based_recommendations(movie_title, movies)
        st.dataframe(recommendations)

with tab2:
    st.subheader("Collaborative Filtering")
    user_id = st.number_input("Enter User ID:", min_value=1, max_value=int(ratings['userId'].max()), value=1)
    if st.button("Recommend for User"):
        recommendations = collaborative_recommendations(user_id, ratings, movies)
        st.dataframe(recommendations)
