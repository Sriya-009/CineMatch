"""Sparse, content-based movie recommendation model."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import load_all_datasets
from .preprocessing import prepare_movie_features


class ContentBasedRecommender:
    """Recommend movies using TF-IDF features and cosine similarity."""

    def __init__(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame | None = None,
        ratings: pd.DataFrame | None = None,
    ):
        self.movies = prepare_movie_features(movies, tags)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.feature_matrix = self.vectorizer.fit_transform(self.movies["feature_text"])
        self.ratings = self._prepare_ratings(ratings)
        self._movie_positions = pd.Series(
            self.movies.index, index=self.movies["movieId"]
        )
        self._rating_matrix: dict[int, dict[int, float]] = {}
        self._item_rating_matrix = None
        if self.ratings is not None:
            self._build_rating_matrix()

    @staticmethod
    def _prepare_ratings(ratings: pd.DataFrame | None) -> pd.DataFrame | None:
        if ratings is None:
            return None
        required_columns = {"userId", "movieId", "rating"}
        if not required_columns.issubset(ratings.columns):
            raise ValueError(
                "Ratings data must contain userId, movieId, and rating columns."
            )
        if ratings.empty:
            return ratings.copy()
        prepared = ratings[list(required_columns)].copy()
        if prepared.isna().any().any():
            raise ValueError("Ratings data cannot contain missing values.")
        if not all(pd.api.types.is_numeric_dtype(prepared[column]) for column in prepared):
            raise ValueError("Ratings userId, movieId, and rating values must be numeric.")
        if ((prepared["rating"] < 0.5) | (prepared["rating"] > 5.0)).any():
            raise ValueError("Ratings must be between 0.5 and 5.0.")
        return prepared

    def _build_rating_matrix(self) -> None:
        assert self.ratings is not None
        known_ratings = self.ratings[
            self.ratings["movieId"].isin(self.movies["movieId"])
        ]
        self._rating_matrix = {
            int(user_id): group.set_index("movieId")["rating"].to_dict()
            for user_id, group in known_ratings.groupby("userId")
        }
        if known_ratings.empty:
            return
        user_positions = {
            user_id: position
            for position, user_id in enumerate(known_ratings["userId"].unique())
        }
        rows = known_ratings["movieId"].map(self._movie_positions).to_numpy()
        columns = known_ratings["userId"].map(user_positions).to_numpy()
        self._item_rating_matrix = csr_matrix(
            (
                known_ratings["rating"].to_numpy(),
                (rows, columns),
            ),
            shape=(len(self.movies), len(user_positions)),
        )

    def _movie_index(self, title: str) -> int:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Movie title must be a non-empty string.")
        matches = self.movies[
            self.movies["title"].str.casefold() == title.strip().casefold()
        ].index.tolist()
        if not matches:
            raise ValueError(f"Movie title was not found: '{title}'.")
        if len(matches) > 1:
            raise ValueError(
                f"Movie title is ambiguous because it appears {len(matches)} times: "
                f"'{title}'. Use a unique title."
            )
        return matches[0]

    def recommend(self, title: str, n: int = 10) -> pd.DataFrame:
        """Return up to ``n`` similar movies, excluding the selected movie."""
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("Recommendation count must be a positive integer.")
        selected_index = self._movie_index(title)
        selected_position = self.movies.index.get_loc(selected_index)
        similarities = cosine_similarity(
            self.feature_matrix[selected_position], self.feature_matrix
        ).ravel()
        similarities[selected_position] = -1.0

        candidate_count = min(n, max(len(self.movies) - 1, 0))
        ranked_positions = similarities.argsort()[::-1][:candidate_count]
        recommendations = self.movies.iloc[ranked_positions].copy()
        recommendations["similarity_score"] = similarities[ranked_positions]
        return recommendations[
            ["movieId", "title", "genres", "release_year", "similarity_score"]
        ].reset_index(drop=True)

    @staticmethod
    def _validate_user_id(user_id: int) -> int:
        if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
            raise ValueError("user_id must be an integer.")
        if user_id <= 0:
            raise ValueError("user_id must be a positive integer.")
        return int(user_id)

    @staticmethod
    def _normalise_preferences(
        genre_preferences: str | Iterable[str] | None,
    ) -> set[str]:
        if genre_preferences is None:
            return set()
        if isinstance(genre_preferences, str):
            values = genre_preferences.replace("|", ",").split(",")
        else:
            values = genre_preferences
        return {str(value).strip().casefold() for value in values if str(value).strip()}

    def _genre_scores(self, preferences: set[str]) -> np.ndarray:
        scores = np.zeros(len(self.movies), dtype=float)
        if not preferences:
            return scores
        for position, genres in enumerate(self.movies["genres"]):
            movie_genres = set(genres.split())
            scores[position] = len(movie_genres & preferences) / len(preferences)
        return scores

    @staticmethod
    def _scale_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        minimum, maximum = scores.min(), scores.max()
        if maximum == minimum:
            return np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        genre_preferences: str | Iterable[str] | None = None,
        min_positive_ratings: int = 2,
    ) -> pd.DataFrame:
        """Recommend unrated movies using ratings, content, and genre signals.

        For users with at least ``min_positive_ratings`` ratings of 4.0 or more,
        the score is ``0.60 * collaborative + 0.30 * content + 0.10 * genre``.
        With less history, collaborative scoring is unavailable and the content
        and genre signals are used as a cold-start fallback.
        """
        user_id = self._validate_user_id(user_id)
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("Recommendation count must be a positive integer.")
        if (
            not isinstance(min_positive_ratings, int)
            or isinstance(min_positive_ratings, bool)
            or min_positive_ratings <= 0
        ):
            raise ValueError("min_positive_ratings must be a positive integer.")
        if self.ratings is None:
            raise ValueError("Ratings data is required for personalized recommendations.")

        preferences = self._normalise_preferences(genre_preferences)
        user_history = self._rating_matrix.get(user_id, {})
        rated_movie_ids = set(user_history)
        positive_history = {
            movie_id: rating
            for movie_id, rating in user_history.items()
            if rating >= 4.0 and movie_id in self._movie_positions
        }
        if not positive_history and not preferences:
            if user_id not in self._rating_matrix:
                raise ValueError(f"Unknown user_id: {user_id}.")
            raise ValueError(
                f"User {user_id} has no positive ratings or genre preferences to use."
            )

        candidate_mask = ~self.movies["movieId"].isin(rated_movie_ids)
        candidate_positions = np.flatnonzero(candidate_mask.to_numpy())
        if len(candidate_positions) == 0:
            return self.movies.iloc[[]].assign(
                collaborative_score=pd.Series(dtype=float),
                content_score=pd.Series(dtype=float),
                genre_score=pd.Series(dtype=float),
                personalized_score=pd.Series(dtype=float),
            )

        content_scores = np.zeros(len(self.movies), dtype=float)
        collaborative_scores = np.zeros(len(self.movies), dtype=float)
        if positive_history:
            liked_positions = [
                self._movie_positions[movie_id] for movie_id in positive_history
            ]
            liked_weights = np.array(
                [positive_history[movie_id] - 3.0 for movie_id in positive_history]
            )
            content_scores = cosine_similarity(
                self.feature_matrix, self.feature_matrix[liked_positions]
            ).dot(liked_weights)
            if self._item_rating_matrix is not None:
                collaborative_scores[candidate_positions] = cosine_similarity(
                    self._item_rating_matrix[candidate_positions],
                    self._item_rating_matrix[liked_positions],
                ).dot(liked_weights)

        genre_scores = self._genre_scores(preferences)
        history_is_sufficient = len(positive_history) >= min_positive_ratings
        if history_is_sufficient:
            final_scores = (
                0.60 * self._scale_scores(collaborative_scores)
                + 0.30 * self._scale_scores(content_scores)
                + 0.10 * genre_scores
            )
        else:
            final_scores = 0.70 * self._scale_scores(content_scores) + 0.30 * genre_scores

        ranked_positions = candidate_positions[
            np.argsort(final_scores[candidate_positions])[::-1]
        ][:n]
        recommendations = self.movies.iloc[ranked_positions].copy()
        recommendations["collaborative_score"] = collaborative_scores[ranked_positions]
        recommendations["content_score"] = content_scores[ranked_positions]
        recommendations["genre_score"] = genre_scores[ranked_positions]
        recommendations["personalized_score"] = final_scores[ranked_positions]
        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "release_year",
                "collaborative_score",
                "content_score",
                "genre_score",
                "personalized_score",
            ]
        ].reset_index(drop=True)


def build_recommender(
    data_directory: str | Path = "data",
) -> ContentBasedRecommender:
    """Load the MovieLens data and construct a content-based recommender."""
    movies, ratings, tags, _ = load_all_datasets(data_directory)
    return ContentBasedRecommender(movies, tags, ratings)
