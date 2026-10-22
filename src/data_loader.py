"""Utilities for loading and validating MovieLens CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

MOVIE_COLUMNS: Final[frozenset[str]] = frozenset({"movieId", "title", "genres"})
RATING_COLUMNS: Final[frozenset[str]] = frozenset(
    {"userId", "movieId", "rating", "timestamp"}
)
TAG_COLUMNS: Final[frozenset[str]] = frozenset(
    {"userId", "movieId", "tag", "timestamp"}
)
LINK_COLUMNS: Final[frozenset[str]] = frozenset({"movieId", "imdbId", "tmdbId"})


class DataValidationError(ValueError):
    """Raised when a MovieLens file is missing or has invalid data."""


def _read_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    """Read a CSV file and convert common file errors into clear messages."""
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset was not found at '{path}'. "
            "Add the MovieLens CSV file to the data directory."
        )
    if not path.is_file():
        raise DataValidationError(f"{dataset_name} path is not a file: '{path}'.")

    try:
        dataframe = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DataValidationError(
            f"Could not read the {dataset_name} dataset at '{path}': {error}"
        ) from error

    if dataframe.empty:
        raise DataValidationError(f"The {dataset_name} dataset is empty: '{path}'.")
    return dataframe


def _validate_columns(
    dataframe: pd.DataFrame, required_columns: frozenset[str], dataset_name: str
) -> None:
    """Ensure a dataset includes every column required by the application."""
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise DataValidationError(
            f"The {dataset_name} dataset is missing required column(s): {columns}."
        )


def _validate_no_missing(
    dataframe: pd.DataFrame, columns: frozenset[str], dataset_name: str
) -> None:
    """Reject missing values in fields needed by recommendation algorithms."""
    missing_counts = dataframe[list(columns)].isna().sum()
    invalid_columns = sorted(
        column for column, missing_count in missing_counts.items() if missing_count
    )
    if invalid_columns:
        columns_text = ", ".join(invalid_columns)
        raise DataValidationError(
            f"The {dataset_name} dataset contains missing values in: {columns_text}."
        )


def load_movies(path: str | Path) -> pd.DataFrame:
    """Load and validate a MovieLens movies file.

    Extra columns are preserved for compatibility with extended MovieLens files.
    """
    movie_path = Path(path)
    movies = _read_csv(movie_path, "movies")
    _validate_columns(movies, MOVIE_COLUMNS, "movies")
    _validate_no_missing(movies, MOVIE_COLUMNS, "movies")

    if not pd.api.types.is_numeric_dtype(movies["movieId"]):
        raise DataValidationError("The movies 'movieId' column must contain numbers.")
    if movies["movieId"].duplicated().any():
        raise DataValidationError("The movies 'movieId' column must contain unique IDs.")
    if (movies["title"].astype("string").str.strip() == "").any():
        raise DataValidationError("The movies 'title' column cannot contain blank titles.")
    return movies


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Load and validate a MovieLens ratings file.

    Extra columns are preserved, while the required rating fields are checked for
    numeric values and a valid MovieLens rating range.
    """
    rating_path = Path(path)
    ratings = _read_csv(rating_path, "ratings")
    _validate_columns(ratings, RATING_COLUMNS, "ratings")
    _validate_no_missing(ratings, RATING_COLUMNS, "ratings")

    numeric_columns = ("userId", "movieId", "rating", "timestamp")
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(ratings[column]):
            raise DataValidationError(
                f"The ratings '{column}' column must contain numbers."
            )
    if ((ratings["rating"] < 0.5) | (ratings["rating"] > 5.0)).any():
        raise DataValidationError("Ratings must be between 0.5 and 5.0.")
    return ratings


def load_tags(path: str | Path) -> pd.DataFrame:
    """Load and validate a MovieLens tags file."""
    tag_path = Path(path)
    tags = _read_csv(tag_path, "tags")
    _validate_columns(tags, TAG_COLUMNS, "tags")
    _validate_no_missing(tags, TAG_COLUMNS, "tags")

    for column in ("userId", "movieId", "timestamp"):
        if not pd.api.types.is_numeric_dtype(tags[column]):
            raise DataValidationError(f"The tags '{column}' column must contain numbers.")
    if (tags["tag"].astype("string").str.strip() == "").any():
        raise DataValidationError("The tags 'tag' column cannot contain blank values.")
    return tags


def load_links(path: str | Path) -> pd.DataFrame:
    """Load and validate a MovieLens links file."""
    link_path = Path(path)
    links = _read_csv(link_path, "links")
    _validate_columns(links, LINK_COLUMNS, "links")

    if links["movieId"].isna().any() or not pd.api.types.is_numeric_dtype(
        links["movieId"]
    ):
        raise DataValidationError("The links 'movieId' column must contain numbers.")
    if links["movieId"].duplicated().any():
        raise DataValidationError("The links 'movieId' column must contain unique IDs.")
    for column in ("imdbId", "tmdbId"):
        non_null_values = links[column].dropna()
        if not pd.api.types.is_numeric_dtype(non_null_values):
            raise DataValidationError(f"The links '{column}' column must contain numbers.")
    return links


def load_datasets(
    data_directory: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both MovieLens datasets from a shared directory."""
    directory = Path(data_directory)
    return load_movies(directory / "movies.csv"), load_ratings(directory / "ratings.csv")


def load_all_datasets(
    data_directory: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load movies, ratings, tags, and links from a shared directory."""
    directory = Path(data_directory)
    return (
        load_movies(directory / "movies.csv"),
        load_ratings(directory / "ratings.csv"),
        load_tags(directory / "tags.csv"),
        load_links(directory / "links.csv"),
    )
