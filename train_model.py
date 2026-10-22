"""Build the content-based model and print a sample recommendation list."""

from __future__ import annotations

from pathlib import Path

from src.recommender import build_recommender


def main() -> None:
    """Train the in-memory model and display recommendations for one movie."""
    data_directory = Path(__file__).resolve().parent / "data"
    recommender = build_recommender(data_directory)
    selected_title = recommender.movies.iloc[0]["title"]
    recommendations = recommender.recommend(selected_title, n=10)
    print(f"Recommendations similar to: {selected_title}")
    print(recommendations.to_string(index=False))

    personalized = recommender.recommend_for_user(
        user_id=1, n=10, genre_preferences=["adventure", "animation"]
    )
    print("\nPersonalized recommendations for user 1")
    print(personalized.to_string(index=False))


if __name__ == "__main__":
    main()
