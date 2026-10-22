"""Data preparation helpers for content-based movie recommendations."""

from __future__ import annotations

import re

import pandas as pd

YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def clean_text(value: object) -> str:
    """Return normalized text suitable for feature construction."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("|", " ").split()).strip().lower()


def extract_release_year(title: object) -> int | None:
    """Extract a trailing four-digit release year from a movie title."""
    if pd.isna(title):
        return None
    match = YEAR_PATTERN.search(str(title).strip())
    return int(match.group(1)) if match else None


def prepare_movie_features(
    movies: pd.DataFrame, tags: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Clean movie metadata and combine genres with aggregated user tags.

    The returned rows retain one row per input movie, including duplicate titles.
    """
    required_columns = {"movieId", "title", "genres"}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        raise ValueError(
            "Movies data is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
    if movies.empty:
        raise ValueError("Movies data cannot be empty.")

    prepared = movies.copy()
    prepared["title"] = prepared["title"].map(str).str.strip()
    prepared["genres"] = prepared["genres"].map(clean_text)
    prepared["release_year"] = prepared["title"].map(extract_release_year)

    prepared["tags"] = ""
    if tags is not None:
        if not {"movieId", "tag"}.issubset(tags.columns):
            raise ValueError("Tags data must contain 'movieId' and 'tag' columns.")
        tag_text = tags.copy()
        tag_text["tag"] = tag_text["tag"].map(clean_text)
        tag_text = tag_text[tag_text["tag"] != ""]
        grouped_tags = tag_text.groupby("movieId", sort=False)["tag"].agg(" ".join)
        prepared["tags"] = prepared["movieId"].map(grouped_tags).fillna("")

    prepared["feature_text"] = (
        prepared["genres"] + " " + prepared["tags"]
    ).str.strip()
    prepared.loc[prepared["feature_text"] == "", "feature_text"] = "unknown"
    return prepared
