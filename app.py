from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

user_item = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_item_filled = user_item.fillna(0)

item_similarity = cosine_similarity(user_item_filled.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_filled.columns,
    columns=user_item_filled.columns
)

movie_titles = movies.set_index("movieId")["title"]


def find_movie(title_part):
    result = movies[movies["title"].str.contains(title_part, case=False, na=False)]
    return result[["movieId", "title"]].head(10)


def get_similar_movies(movie_id, top_n=5):
    similar_scores = item_similarity_df[movie_id].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_id)

    result = pd.DataFrame({
        "movieId": similar_scores.index,
        "similarity": similar_scores.values
    }).head(top_n)

    result["title"] = result["movieId"].map(movie_titles)
    return result[["title", "similarity"]]


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    matches = None
    query = ""

    if request.method == "POST":
        query = request.form.get("movie_title", "").strip()

        if query:
            matches = find_movie(query)

            if not matches.empty:
                movie_id = matches.iloc[0]["movieId"]
                recommendations = get_similar_movies(movie_id).to_dict(orient="records")

    return render_template(
        "index.html",
        recommendations=recommendations,
        matches=matches.to_dict(orient="records") if matches is not None and not matches.empty else None,
        query=query
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)