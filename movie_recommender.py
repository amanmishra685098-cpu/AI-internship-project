"""
Movie Recommendation System
-----------------------------
A simple, self-contained recommender that suggests movies similar to one
you already like, combining:

  1. Content-based filtering  - using genres + description via TF-IDF and
     cosine similarity.
  2. Collaborative filtering  - using a small user-ratings matrix and
     item-item cosine similarity.

The script ships with a small built-in dataset so it runs immediately with
no external files or internet connection needed. To use your own data,
replace `build_movie_catalog()` and `build_ratings_matrix()` with loaders
for your own CSV files (e.g. the MovieLens dataset).

Requirements:
    pip install pandas scikit-learn numpy

Run:
    python movie_recommender.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def build_movie_catalog() -> pd.DataFrame:
    """A small built-in movie catalog with genres and short descriptions."""
    data = [
        ("Inception", "Sci-Fi Thriller", "A thief who steals secrets through dream-sharing technology is given a chance to have his criminal history erased."),
        ("Interstellar", "Sci-Fi Drama", "A team of explorers travel through a wormhole in space to ensure humanity's survival."),
        ("The Dark Knight", "Action Crime Thriller", "Batman faces the Joker, a criminal mastermind who wants to plunge Gotham into anarchy."),
        ("The Prestige", "Mystery Thriller", "Two magicians engage in a bitter rivalry to create the ultimate stage illusion."),
        ("Memento", "Mystery Thriller", "A man with short-term memory loss uses notes and tattoos to hunt for his wife's killer."),
        ("The Matrix", "Sci-Fi Action", "A hacker learns the true nature of his reality and his role in the war against its controllers."),
        ("Toy Story", "Animation Family Comedy", "A cowboy doll feels threatened when a new spaceman action figure becomes his owner's favorite toy."),
        ("Finding Nemo", "Animation Family Adventure", "A clownfish searches the ocean for his missing son with the help of a forgetful blue tang."),
        ("Up", "Animation Family Adventure", "An elderly widower and a young boy travel to South America in a house lifted by balloons."),
        ("The Notebook", "Romance Drama", "A poor young man and a rich young woman fall in love and are torn apart by social class."),
        ("La La Land", "Romance Musical Drama", "A jazz musician and an aspiring actress fall in love while pursuing their dreams in Los Angeles."),
        ("Pride and Prejudice", "Romance Drama", "Elizabeth Bennet navigates love and society expectations in early 19th century England."),
        ("John Wick", "Action Thriller", "A retired hitman seeks vengeance against the gangsters who took everything from him."),
        ("Mad Max: Fury Road", "Action Adventure Sci-Fi", "In a post-apocalyptic wasteland, a woman rebels against a tyrant in search of her homeland."),
        ("Gladiator", "Action Drama History", "A betrayed Roman general seeks revenge against the corrupt emperor who murdered his family."),
        ("Get Out", "Horror Thriller Mystery", "A young man uncovers a disturbing secret when he visits his girlfriend's family estate."),
        ("A Quiet Place", "Horror Thriller Sci-Fi", "A family must live in silence to avoid attracting creatures that hunt by sound."),
        ("Hereditary", "Horror Mystery Drama", "A family unravels dark secrets after the death of their secretive grandmother."),
        ("The Hangover", "Comedy", "Three friends wake up with no memory of the previous night and must find their missing groom."),
        ("Superbad", "Comedy", "Two co-dependent best friends try to make the most of their final days of high school."),
    ]
    return pd.DataFrame(data, columns=["title", "genres", "description"])


def build_ratings_matrix(movie_titles) -> pd.DataFrame:
    """
    A small synthetic user-item ratings matrix (rows = users, columns = movies).
    0 means "not rated". In a real project, load this from a ratings CSV
    (e.g. MovieLens `ratings.csv` pivoted to a user x movie matrix).
    """
    rng = np.random.default_rng(42)
    n_users = 15
    ratings = rng.integers(0, 6, size=(n_users, len(movie_titles)))  # 0-5, 0 = unrated
    # Make it sparse/realistic: randomly zero out ~55% of entries
    mask = rng.random((n_users, len(movie_titles))) < 0.55
    ratings[mask] = 0
    return pd.DataFrame(ratings, columns=movie_titles, index=[f"user_{i+1}" for i in range(n_users)])


# ---------------------------------------------------------------------------
# 2. Content-based filtering
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        # Combine genres (weighted x2) and description into one text field
        combined_text = (self.catalog["genres"] + " " + self.catalog["genres"] + " " + self.catalog["description"])
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(combined_text)
        self.similarity = cosine_similarity(tfidf_matrix)
        self.title_to_idx = {t: i for i, t in enumerate(self.catalog["title"])}

    def recommend(self, title: str, top_n: int = 5):
        if title not in self.title_to_idx:
            raise ValueError(f"'{title}' not found in catalog.")
        idx = self.title_to_idx[title]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]
        return [(self.catalog.iloc[i]["title"], round(score, 3)) for i, score in scores]


# ---------------------------------------------------------------------------
# 3. Collaborative filtering (item-based)
# ---------------------------------------------------------------------------

class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings
        # Item-item similarity based on user rating patterns
        item_similarity = cosine_similarity(ratings.T)
        self.similarity_df = pd.DataFrame(item_similarity, index=ratings.columns, columns=ratings.columns)

    def recommend(self, title: str, top_n: int = 5):
        if title not in self.similarity_df.columns:
            raise ValueError(f"'{title}' not found in ratings matrix.")
        scores = self.similarity_df[title].drop(title).sort_values(ascending=False)
        return list(zip(scores.index[:top_n], scores.round(3).values[:top_n]))

    def recommend_for_user(self, user: str, top_n: int = 5):
        """Suggest unrated movies for a user, weighted by similarity to movies they rated highly."""
        user_ratings = self.ratings.loc[user]
        rated = user_ratings[user_ratings > 0]
        if rated.empty:
            return []

        scores = pd.Series(0.0, index=self.ratings.columns)
        for movie, rating in rated.items():
            scores += self.similarity_df[movie] * rating
        scores = scores.drop(rated.index)  # exclude already-rated movies
        scores = scores.sort_values(ascending=False)
        return list(zip(scores.index[:top_n], scores.round(3).values[:top_n]))


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

def main():
    catalog = build_movie_catalog()
    ratings = build_ratings_matrix(catalog["title"].tolist())

    print("=" * 70)
    print("CONTENT-BASED RECOMMENDATIONS (based on genre + description)")
    print("=" * 70)
    content_rec = ContentBasedRecommender(catalog)
    for seed_movie in ["Inception", "Toy Story", "John Wick"]:
        print(f"\nBecause you liked '{seed_movie}':")
        for title, score in content_rec.recommend(seed_movie, top_n=5):
            print(f"  - {title}  (similarity: {score})")

    print("\n" + "=" * 70)
    print("COLLABORATIVE FILTERING RECOMMENDATIONS (based on user rating patterns)")
    print("=" * 70)
    collab_rec = CollaborativeRecommender(ratings)
    for seed_movie in ["Inception", "Toy Story"]:
        print(f"\nUsers who rated '{seed_movie}' highly also rated:")
        for title, score in collab_rec.recommend(seed_movie, top_n=5):
            print(f"  - {title}  (similarity: {score})")

    print("\n" + "=" * 70)
    print("PERSONALIZED RECOMMENDATIONS FOR A SPECIFIC USER")
    print("=" * 70)
    for user in ["user_1", "user_5"]:
        rated = ratings.loc[user]
        rated = rated[rated > 0]
        print(f"\n{user} rated: {dict(rated)}")
        print(f"Recommended for {user}:")
        for title, score in collab_rec.recommend_for_user(user, top_n=5):
            print(f"  - {title}  (score: {score})")


if __name__ == "__main__":
    main()
