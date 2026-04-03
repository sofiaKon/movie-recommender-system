from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

user_item = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)  # movie(colomns) - user(rows) matrix

user_item_filled = user_item.fillna(0)  # preference matrix


item_similarity = cosine_similarity(user_item_filled.T)
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_filled.columns,
    columns=user_item_filled.columns
)

# dictionary of movieId and title
movie_titles = movies.set_index("movieId")["title"]


def find_movie(title_part):
    result = movies[movies["title"].str.contains(
        title_part, case=False, na=False)]
    return result[["movieId", "title"]].head(10)


def get_similar_movies(movie_id, top_n=5):
    similar_scores = item_similarity_df[movie_id].sort_values(ascending=False)

    similar_scores = similar_scores.drop(movie_id)

    result = pd.DataFrame({
        "movieId": similar_scores.index,
        "similarity": similar_scores.values
    }).head(top_n)

    result["title"] = result["movieId"].map(movie_titles)

    return result[["movieId", "title", "similarity"]]


def recommend_movies_for_user(user_id, top_n=5, min_similarity=0.1, top_k=20): #Method to recommend movies for a user based on really watched movies
    user_ratings = user_item_filled.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]
    unrated_movies = user_ratings[user_ratings == 0].index

    predictions = {}

    for movie_id in unrated_movies:
        similarities = item_similarity_df.loc[movie_id, rated_movies.index]
        similarities = similarities[similarities > min_similarity].sort_values(ascending=False).head(top_k)

        if len(similarities) == 0:
            continue

        relevant_ratings = rated_movies[similarities.index]

        weighted_sum = (similarities * relevant_ratings).sum()
        sim_sum = similarities.sum()

        if sim_sum > 0:
            predictions[movie_id] = weighted_sum / sim_sum

    predicted_scores = pd.Series(predictions).sort_values(ascending=False).head(top_n)

    result = pd.DataFrame({
        "movieId": predicted_scores.index,
        "predicted_rating": predicted_scores.values
    })

    result["title"] = result["movieId"].map(movie_titles)

    return result[["movieId", "title", "predicted_rating"]]