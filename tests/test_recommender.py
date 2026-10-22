from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataValidationError, load_all_datasets, load_datasets
from src.recommender import ContentBasedRecommender


def movie_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Space Adventure (2000)", "genres": "Action|Sci-Fi"},
            {"movieId": 2, "title": "Space Mission (2001)", "genres": "Action|Sci-Fi"},
            {"movieId": 3, "title": "Quiet Romance (2002)", "genres": "Romance"},
            {"movieId": 4, "title": "Funny Space (2003)", "genres": "Comedy|Sci-Fi"},
        ]
    )


def tag_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "tag": "spaceship", "timestamp": 1},
            {"userId": 1, "movieId": 2, "tag": "spaceship", "timestamp": 1},
        ]
    )


def test_dataset_loading(tmp_path: Path) -> None:
    movies = movie_data()
    ratings = pd.DataFrame(
        [{"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1}]
    )
    tags = tag_data()
    links = pd.DataFrame([{"movieId": 1, "imdbId": 1, "tmdbId": 1}])
    movies.to_csv(tmp_path / "movies.csv", index=False)
    ratings.to_csv(tmp_path / "ratings.csv", index=False)
    tags.to_csv(tmp_path / "tags.csv", index=False)
    links.to_csv(tmp_path / "links.csv", index=False)

    loaded_movies, loaded_ratings = load_datasets(tmp_path)
    loaded_all = load_all_datasets(tmp_path)
    assert len(loaded_movies) == 4
    assert len(loaded_ratings) == 1
    assert all(len(frame) > 0 for frame in loaded_all)


def test_successful_recommendations_and_exclusion() -> None:
    recommender = ContentBasedRecommender(movie_data(), tag_data())
    results = recommender.recommend("Space Adventure (2000)", n=10)
    assert 0 < len(results) <= 3
    assert "Space Adventure (2000)" not in results["title"].tolist()
    assert {"title", "genres", "release_year", "similarity_score"} <= set(results)


def test_unknown_title_and_invalid_inputs() -> None:
    recommender = ContentBasedRecommender(movie_data(), tag_data())
    with pytest.raises(ValueError, match="not found"):
        recommender.recommend("Unknown Movie")
    with pytest.raises(ValueError, match="non-empty"):
        recommender.recommend("", n=1)
    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend("Space Adventure (2000)", n=0)


def test_missing_tags_are_supported() -> None:
    recommender = ContentBasedRecommender(movie_data(), None)
    results = recommender.recommend("Quiet Romance (2002)", n=2)
    assert len(results) == 2


def test_empty_movies_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ContentBasedRecommender(pd.DataFrame(columns=["movieId", "title", "genres"]))


def test_duplicate_titles_are_reported() -> None:
    movies = pd.concat([movie_data(), movie_data().iloc[[0]]], ignore_index=True)
    recommender = ContentBasedRecommender(movies)
    with pytest.raises(ValueError, match="ambiguous"):
        recommender.recommend("Space Adventure (2000)")


def test_invalid_rating_data_is_rejected(tmp_path: Path) -> None:
    movies = movie_data().iloc[[0]]
    ratings = pd.DataFrame(
        [{"userId": 1, "movieId": 1, "rating": 6.0, "timestamp": 1}]
    )
    movies.to_csv(tmp_path / "movies.csv", index=False)
    ratings.to_csv(tmp_path / "ratings.csv", index=False)
    with pytest.raises(DataValidationError, match="between 0.5 and 5.0"):
        load_datasets(tmp_path)


def rating_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0},
            {"userId": 1, "movieId": 3, "rating": 1.0},
            {"userId": 2, "movieId": 2, "rating": 4.0},
        ]
    )


def test_personalized_recommendations_use_history_and_exclude_rated_movies() -> None:
    recommender = ContentBasedRecommender(movie_data(), tag_data(), rating_data())
    results = recommender.recommend_for_user(1, n=10)
    assert len(results) == 2
    assert not {1, 3}.intersection(results["movieId"])
    assert {
        "collaborative_score",
        "content_score",
        "genre_score",
        "personalized_score",
    } <= set(results)


def test_genre_preferences_support_cold_start_user() -> None:
    recommender = ContentBasedRecommender(movie_data(), ratings=rating_data())
    results = recommender.recommend_for_user(
        999, n=1, genre_preferences=["romance"]
    )
    assert results.iloc[0]["movieId"] == 3


def test_unknown_user_without_preferences_and_invalid_user_id() -> None:
    recommender = ContentBasedRecommender(movie_data(), ratings=rating_data())
    with pytest.raises(ValueError, match="Unknown user_id"):
        recommender.recommend_for_user(999)
    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend_for_user(0)
    with pytest.raises(ValueError, match="integer"):
        recommender.recommend_for_user("1")


def test_insufficient_history_falls_back_to_content() -> None:
    ratings = pd.DataFrame([{"userId": 5, "movieId": 1, "rating": 5.0}])
    recommender = ContentBasedRecommender(movie_data(), ratings=ratings)
    results = recommender.recommend_for_user(5, n=2)
    assert len(results) == 2
    assert not set(results["movieId"]).intersection({1})
