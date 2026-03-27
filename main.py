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

print("Размер user-item matrix:", user_item_filled.shape)
print("Размер матрицы похожести:", item_similarity_df.shape)
