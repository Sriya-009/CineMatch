"""CineMatch Streamlit application."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from src.data_loader import DataValidationError, load_all_datasets
from src.preprocessing import extract_release_year
from src.recommender import ContentBasedRecommender

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
PAGES = ["Home", "Discover Movies", "Similar Movies", "Personalized For You", "Rate Movies"]
TMDB_API_DEFAULT = "https://api.themoviedb.org/3"
TMDB_IMAGE_DEFAULT = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="CineMatch | Find your next favorite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Apply CineMatch's dark cinematic design system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
        :root { color-scheme: dark; }
        .stApp {
            background: #08090d;
            color: #f3f4f6;
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 80% -10%, #1c1930 0, #0d1019 36%, #08090d 72%);
        }
        [data-testid="stSidebar"] {
            background: #0b0c12;
            border-right: 1px solid #242633;
        }
        [data-testid="stSidebar"] * { color: #d9dbe5; }
        [data-testid="stSidebar"] .stRadio label {
            border-radius: 10px;
            padding: .55rem .7rem;
            transition: background .2s ease;
        }
        [data-testid="stSidebar"] .stRadio label:hover { background: #1b1b28; }
        h1, h2, h3 { font-family: 'Playfair Display', serif; letter-spacing: -.025em; }
        h1 { font-size: clamp(2.2rem, 5vw, 4.2rem) !important; line-height: 1.02 !important; }
        h2 { font-size: 2rem !important; }
        .brand-mark { color: #e85d75; font-size: 1.6rem; font-weight: 700; letter-spacing: -.04em; }
        .brand-copy { color: #878b9d; font-size: .78rem; line-height: 1.45; }
        .eyebrow { color: #ef6b7e; font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
        .muted { color: #999eae; }
        .hero {
            min-height: 300px;
            padding: 3.4rem 3.4rem 2.8rem;
            border: 1px solid #2c2b43;
            border-radius: 24px;
            background: linear-gradient(110deg, rgba(28, 27, 49, .97), rgba(14, 17, 27, .74)),
                        radial-gradient(circle at 90% 30%, #47203a, transparent 42%);
            box-shadow: 0 24px 70px rgba(0, 0, 0, .36);
            margin-bottom: 2rem;
        }
        .hero p { max-width: 620px; color: #aeb2c2; font-size: 1.03rem; line-height: 1.7; }
        .section-kicker { color: #878b9d; font-size: .82rem; letter-spacing: .12em; text-transform: uppercase; }
        .movie-card {
            overflow: hidden;
            min-height: 380px;
            border: 1px solid #292b39;
            border-radius: 16px;
            background: #12141d;
            box-shadow: 0 10px 28px rgba(0, 0, 0, .22);
            margin-bottom: 1.2rem;
        }
        .poster-frame {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 235px;
            padding: 1rem;
            background: linear-gradient(145deg, #252238, #141520);
            color: #b6aabd;
            text-align: center;
        }
        [data-testid="stImage"] img {
            width: 100%;
            height: 235px;
            object-fit: cover;
            border-radius: 10px;
        }
        .poster-frame img { width: 100%; height: 100%; object-fit: cover; border-radius: 10px; }
        .poster-placeholder { color: #a9a1b7; font-family: 'Playfair Display', serif; font-size: 1.15rem; }
        .movie-card-body { padding: 1rem 1.05rem 1.1rem; }
        .movie-card h3 { font-family: 'DM Sans', sans-serif; font-size: 1rem; margin: 0 0 .45rem; }
        .movie-card p { color: #9ea4b6; font-size: .82rem; line-height: 1.45; margin: .24rem 0; }
        .movie-card .score { color: #ef6b7e; font-weight: 700; }
        .pill {
            display: inline-block;
            padding: .2rem .52rem;
            border: 1px solid #393348;
            border-radius: 999px;
            color: #c1b7ce;
            font-size: .7rem;
        }
        .stButton > button, .stFormSubmitButton > button {
            border: 1px solid #e85d75;
            border-radius: 10px;
            background: #e85d75;
            color: #160b10;
            font-weight: 700;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover { border-color: #ff8798; background: #ff8798; }
        div[data-testid="stMetric"] {
            padding: .9rem 1rem;
            border: 1px solid #282a38;
            border-radius: 12px;
            background: #11131b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_app_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load validated MovieLens data once per session."""
    return load_all_datasets(DATA_DIRECTORY)


@st.cache_resource(show_spinner=False)
def build_app_recommender(
    movies: pd.DataFrame, tags: pd.DataFrame, ratings: pd.DataFrame
) -> ContentBasedRecommender:
    """Build and cache the existing recommendation model."""
    return ContentBasedRecommender(movies, tags, ratings)


def prepare_display_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Add display-only release years without changing model inputs."""
    display = movies.copy()
    display["release_year"] = display["title"].map(extract_release_year)
    return display


def load_dotenv_file() -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        name, value = name.strip(), value.strip().strip("'\"")
        if name and value and name not in os.environ:
            os.environ[name] = value


def tmdb_api_key() -> str:
    """Resolve the TMDB key without exposing it in UI or logs."""
    load_dotenv_file()
    environment_key = os.environ.get("TMDB_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        secret_key = st.secrets.get("TMDB_API_KEY", "")
    except (FileNotFoundError, KeyError):
        return ""
    return str(secret_key).strip()


def tmdb_api_base_url() -> str:
    """Return the configured TMDB API endpoint."""
    return os.getenv("TMDB_API_BASE_URL", TMDB_API_DEFAULT).rstrip("/")


def tmdb_image_base_url() -> str:
    """Return the configured TMDB image endpoint."""
    return os.getenv("TMDB_IMAGE_BASE_URL", TMDB_IMAGE_DEFAULT).rstrip("/")


def movie_year(movie: pd.Series) -> int | None:
    """Return a reliable release year when the title includes one."""
    value = movie.get("release_year")
    return int(value) if pd.notna(value) else None


@st.cache_data(ttl=900, show_spinner=False)
def lookup_poster_path_by_id(tmdb_id: int, api_key: str) -> str | None:
    """Find and cache a poster path through TMDB's movie-ID endpoint."""
    if tmdb_id <= 0 or not api_key:
        return None
    parameters = urlencode({"api_key": api_key})
    url = f"{tmdb_api_base_url()}/movie/{tmdb_id}?{parameters}"
    request = Request(url, headers={"User-Agent": "CineMatch/1.0"})
    try:
        with urlopen(request, timeout=4) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None
    poster_path = payload.get("poster_path")
    return poster_path if isinstance(poster_path, str) and poster_path else None


@st.cache_data(ttl=900, show_spinner=False)
def lookup_poster_path_by_title(
    title: str, year: int | None, api_key: str
) -> str | None:
    """Find and cache one visible movie's TMDB poster path by title.

    The short TTL also caches failed and missing-poster results briefly, avoiding
    repeated requests while allowing a temporary TMDB outage to recover.
    """
    if not api_key:
        return None
    search_title = re.sub(r"\s+\(\d{4}\)\s*$", "", title).strip()
    if not search_title:
        return None
    parameters = {"api_key": api_key, "query": search_title}
    if year:
        parameters["year"] = str(year)
    url = f"{tmdb_api_base_url()}/search/movie?{urlencode(parameters)}"
    request = Request(url, headers={"User-Agent": "CineMatch/1.0"})
    try:
        with urlopen(request, timeout=4) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None
    results = payload.get("results", [])
    if not isinstance(results, list):
        return None
    if year:
        matching_year = [
            result for result in results
            if str(result.get("release_date", "")).startswith(str(year))
        ]
        results = matching_year or results
    normalized_title = search_title.casefold()
    exact_title = [
        result for result in results
        if str(result.get("title", "")).casefold().strip() == normalized_title
    ]
    result = (exact_title or results)[0] if (exact_title or results) else None
    poster_path = result.get("poster_path") if result else None
    return poster_path if isinstance(poster_path, str) and poster_path else None


def movie_tmdb_id(movie: pd.Series, links: pd.DataFrame | None = None) -> int | None:
    """Return a usable TMDB ID from the movie row or MovieLens links data."""
    value = movie.get("tmdbId")
    if (value is None or pd.isna(value)) and links is not None:
        matching = links.loc[links["movieId"] == movie.get("movieId"), "tmdbId"]
        value = matching.iloc[0] if not matching.empty else None
    try:
        tmdb_id = int(value)
    except (TypeError, ValueError):
        return None
    return tmdb_id if tmdb_id > 0 else None


def poster_url(movie: pd.Series, links: pd.DataFrame | None = None) -> str | None:
    """Return a verified TMDB image URL for one visible movie."""
    api_key = tmdb_api_key()
    if not api_key:
        return None
    tmdb_id = movie_tmdb_id(movie, links)
    if tmdb_id is not None:
        path = lookup_poster_path_by_id(tmdb_id, api_key)
    else:
        path = lookup_poster_path_by_title(
            str(movie.get("title", "")), movie_year(movie), api_key
        )
    if not path:
        return None
    return f"{tmdb_image_base_url()}/{path.lstrip('/')}"


def poster_placeholder(title: str) -> str:
    """Create a safe local SVG fallback for unavailable poster images."""
    escaped_title = html.escape(title, quote=True)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="750" '
        'viewBox="0 0 500 750"><rect width="500" height="750" fill="#252238"/>'
        '<text x="250" y="370" fill="#b6aabd" font-family="Georgia" '
        'font-size="28" text-anchor="middle">'
        f"{escaped_title}</text></svg>"
    )
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def select_movie(movie_id: object) -> None:
    """Store the selected movie ID for the details view."""
    try:
        st.session_state["selected_movie_id"] = int(movie_id)
    except (TypeError, ValueError):
        st.session_state.pop("selected_movie_id", None)


def update_navigation_state(page: str) -> None:
    """Clear movie details when the user switches to another page."""
    previous_page = st.session_state.get("previous_page")
    if previous_page is not None and previous_page != page:
        st.session_state.pop("selected_movie_id", None)
    st.session_state["previous_page"] = page


def render_movie_card(
    movie: pd.Series,
    score_column: str | None = None,
    links: pd.DataFrame | None = None,
) -> None:
    """Render a consistent responsive movie card with an optional real poster."""
    title = html.escape(str(movie.get("title", "Unknown title")))
    genres = html.escape(str(movie.get("genres", "Unknown genres")).replace("|", " · "))
    year = movie_year(movie)
    year_text = str(year) if year else "Release year unavailable"
    score_text = ""
    if score_column and score_column in movie and pd.notna(movie[score_column]):
        score_text = (
            f'<p class="score">Match score · {float(movie[score_column]):.3f}</p>'
        )
    with st.container(border=True):
        poster = poster_url(movie, links)
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.image(poster_placeholder(str(movie.get("title", "Unknown title"))),
                     use_container_width=True)
        st.markdown(
            f"""
            <div class="movie-card-body">
                <h3>{title}</h3>
                <p>{genres}</p>
                <p>{year_text}</p>
                {score_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "View details",
            key=f"movie-details-{movie.get('movieId')}",
            use_container_width=True,
        ):
            select_movie(movie.get("movieId"))
            st.rerun()


def render_cards(
    movies: pd.DataFrame,
    score_column: str | None = None,
    links: pd.DataFrame | None = None,
) -> None:
    """Render movie data in a responsive three-column grid."""
    if movies.empty:
        st.info("No movies matched those choices.")
        return
    columns = st.columns(3)
    for position, (_, movie) in enumerate(movies.iterrows()):
        with columns[position % 3]:
            render_movie_card(movie, score_column, links)


def render_movie_details(
    movies: pd.DataFrame, links: pd.DataFrame
) -> None:
    """Render details for the movie selected from a card."""
    selected_id = st.session_state.get("selected_movie_id")
    if selected_id is None:
        return
    matches = movies.loc[movies["movieId"] == selected_id]
    if matches.empty:
        st.session_state.pop("selected_movie_id", None)
        st.warning("That movie is no longer available.")
        return

    movie = prepare_display_movies(matches).iloc[0]
    if st.button("← Back to movies", key="movie-details-back"):
        st.session_state.pop("selected_movie_id", None)
        st.rerun()

    st.title(str(movie["title"]))
    poster = poster_url(movie, links)
    detail_columns = st.columns([1, 2])
    with detail_columns[0]:
        st.image(
            poster or poster_placeholder(str(movie["title"])),
            use_container_width=True,
        )
    with detail_columns[1]:
        year = movie_year(movie)
        st.subheader("Movie details")
        st.write(f"**Genre:** {str(movie['genres']).replace('|', ' · ')}")
        st.write(f"**Release year:** {year if year else 'Unavailable'}")
        st.write(f"**MovieLens ID:** {int(movie['movieId'])}")
        tmdb_id = movie_tmdb_id(movie, links)
        if tmdb_id is not None:
            st.write(f"**TMDB ID:** {tmdb_id}")


def genres_from_movies(movies: pd.DataFrame) -> list[str]:
    """Return sorted, displayable genres from the catalog."""
    return sorted(
        {
            genre
            for values in movies["genres"].fillna("").str.split("|")
            for genre in values
            if genre and genre != "(no genres listed)"
        }
    )


def movie_choices(movies: pd.DataFrame, query: str, genre: str) -> pd.DataFrame:
    """Filter catalog titles and genres."""
    filtered = movies
    if query.strip():
        filtered = filtered[
            filtered["title"].str.contains(query.strip(), case=False, na=False, regex=False)
        ]
    if genre != "All genres":
        filtered = filtered[
            filtered["genres"].str.split("|").map(lambda values: genre in values)
        ]
    return filtered


def render_home(
    movies: pd.DataFrame,
    recommender: ContentBasedRecommender,
    links: pd.DataFrame,
) -> None:
    """Render the landing page and real dataset-driven featured rows."""
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">CineMatch · Movie discovery, tuned to you</div>
            <h1>Some stories<br>stay with you.</h1>
            <p>Explore a cinematic catalog through genre and tag similarity, discover
            hidden connections, and shape your next watch with your own preferences.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = st.columns(3)
    metrics[0].metric("Movies to explore", f"{len(movies):,}")
    metrics[1].metric("Genres", f"{len(genres_from_movies(movies))}")
    metrics[2].metric("Recommendation engine", "TF-IDF + ratings")
    featured_title = "Toy Story (1995)"
    if featured_title not in recommender.movies["title"].values:
        featured_title = str(recommender.movies.iloc[0]["title"])
    st.markdown('<div class="section-kicker">A place to start</div>', unsafe_allow_html=True)
    st.subheader(f"Because you liked {featured_title}")
    render_cards(
        recommender.recommend(featured_title, n=6), "similarity_score", links
    )

    display = prepare_display_movies(movies)
    st.markdown('<div class="section-kicker">Curated from the catalog</div>', unsafe_allow_html=True)
    st.subheader("Adventure for tonight")
    adventure = display[display["genres"].str.contains("Adventure", na=False)].head(6)
    render_cards(adventure, links=links)


def render_discover(movies: pd.DataFrame, links: pd.DataFrame) -> None:
    """Render title search and genre discovery."""
    display = prepare_display_movies(movies)
    st.title("Discover Movies")
    st.caption("Search the catalog, filter by genre, and find a new story to follow.")
    search, genre_column = st.columns([2, 1])
    with search:
        query = st.text_input("Search titles", placeholder="Search by title…", label_visibility="collapsed")
    with genre_column:
        genre = st.selectbox("Genre", ["All genres", *genres_from_movies(movies)], label_visibility="collapsed")
    result = movie_choices(display, query, genre)
    st.markdown(f'<div class="section-kicker">{len(result):,} movies found</div>', unsafe_allow_html=True)
    render_cards(result.head(30), links=links)


def render_similar(
    recommender: ContentBasedRecommender, links: pd.DataFrame
) -> None:
    """Render content-based similar movie results."""
    st.title("Similar Movies")
    st.caption("Start with one title. CineMatch finds nearby stories from genres and tags.")
    titles = recommender.movies["title"].tolist()
    default_index = titles.index("Toy Story (1995)") if "Toy Story (1995)" in titles else 0
    selected = st.selectbox("Choose a movie", titles, index=default_index)
    count = st.slider("Number of recommendations", 1, 10, 10)
    try:
        results = recommender.recommend(selected, count)
    except ValueError as error:
        st.error(str(error))
        return
    render_cards(results, "similarity_score", links)


def recommender_with_session_ratings(
    movies: pd.DataFrame, tags: pd.DataFrame, ratings: pd.DataFrame
) -> ContentBasedRecommender:
    """Build a temporary in-memory model including session-only ratings."""
    session_ratings = st.session_state.get("submitted_ratings", {})
    if not session_ratings:
        return build_app_recommender(movies, tags, ratings)
    extra = pd.DataFrame(
        [
            {"userId": user_id, "movieId": movie_id, "rating": score, "timestamp": 0}
            for (user_id, movie_id), score in session_ratings.items()
        ]
    )
    return ContentBasedRecommender(movies, tags, pd.concat([ratings, extra], ignore_index=True))


def render_personalized(
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    ratings: pd.DataFrame,
    links: pd.DataFrame,
) -> None:
    """Render rating-history and genre-based personalized recommendations."""
    st.title("Personalized For You")
    st.caption("Enter a MovieLens user ID or add session ratings to guide your next watch.")
    controls, guidance = st.columns([1, 1])
    with controls:
        user_id = st.number_input("MovieLens user ID", min_value=1, step=1, value=1)
        preferences = st.multiselect("Preferred genres", genres_from_movies(movies))
        count = st.slider("Number of recommendations", 1, 10, 10)
        build = st.button("Curate my watchlist", type="primary")
    with guidance:
        st.info(
            "CineMatch combines collaborative signals from ratings with content "
            "similarity. For new users, select genres or rate movies first."
        )
    if build:
        model = recommender_with_session_ratings(movies, tags, ratings)
        try:
            results = model.recommend_for_user(
                int(user_id), n=count, genre_preferences=preferences
            )
        except ValueError as error:
            st.error(str(error))
            return
        if results.empty:
            st.info("There are no unrated movies left to recommend for this user.")
            return
        st.success("Your watchlist is ready.")
        render_cards(results, "personalized_score", links)


def render_rate_movies(movies: pd.DataFrame) -> None:
    """Collect ratings for this browser session without mutating CSV data."""
    display = prepare_display_movies(movies)
    st.title("Rate Movies")
    st.caption("Try ratings here; they stay in this browser session and do not change MovieLens files.")
    user_id = st.number_input("Your session user ID", min_value=1, step=1, value=1)
    query = st.text_input("Find a movie to rate", placeholder="Search titles…")
    matches = movie_choices(display, query, "All genres").head(50)
    if matches.empty:
        st.info("No titles matched your search.")
        return
    selected = st.multiselect("Select movies", matches["title"].tolist())
    if not selected:
        st.info("Select one or more movies to enter ratings.")
        return
    with st.form("rating_form"):
        entries = {
            title: st.slider(title, 0.5, 5.0, 4.0, 0.5) for title in selected
        }
        submitted = st.form_submit_button("Save session ratings")
    if submitted:
        movie_ids = display.set_index("title")["movieId"]
        stored = st.session_state.setdefault("submitted_ratings", {})
        for title, score in entries.items():
            stored[(int(user_id), int(movie_ids[title]))] = score
        st.success(f"Saved {len(entries)} rating(s) for user {int(user_id)} in this session.")


def main() -> None:
    """Run the CineMatch application."""
    inject_styles()
    st.sidebar.markdown('<div class="brand-mark">CineMatch</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="brand-copy">A smarter way to find<br>your next favorite film.</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Explore", PAGES, key="navigation_page", label_visibility="collapsed"
    )
    update_navigation_state(page)
    st.sidebar.divider()
    if tmdb_api_key():
        st.sidebar.caption("● Poster lookup enabled")
    else:
        st.sidebar.caption("○ Poster lookup unavailable · placeholders enabled")
    try:
        movies, ratings, tags, links = load_app_data()
        recommender = build_app_recommender(movies, tags, ratings)
    except (FileNotFoundError, DataValidationError, ValueError) as error:
        st.error(f"Unable to load the movie data: {error}")
        st.info("Place valid movies.csv, ratings.csv, tags.csv, and links.csv files in data/.")
        st.stop()

    if st.session_state.get("selected_movie_id") is not None:
        render_movie_details(movies, links)
    elif page == "Home":
        render_home(movies, recommender, links)
    elif page == "Discover Movies":
        render_discover(movies, links)
    elif page == "Similar Movies":
        render_similar(recommender, links)
    elif page == "Personalized For You":
        render_personalized(movies, tags, ratings, links)
    else:
        render_rate_movies(movies)


if __name__ == "__main__":
    main()
