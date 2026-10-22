import json
from urllib.parse import parse_qs, urlparse

import pandas as pd

import app


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_id_lookup_returns_poster_path_and_uses_movie_endpoint(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse({"poster_path": "/poster.jpg"})

    monkeypatch.setattr(app, "urlopen", fake_urlopen)
    app.lookup_poster_path_by_id.clear()

    assert app.lookup_poster_path_by_id(862, "test-key") == "/poster.jpg"
    parsed = urlparse(captured["url"])
    assert parsed.path.endswith("/movie/862")
    assert parse_qs(parsed.query)["api_key"] == ["test-key"]
    assert captured["timeout"] == 4


def test_missing_id_poster_is_cached_result(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse({"poster_path": None})

    monkeypatch.setattr(app, "urlopen", fake_urlopen)
    app.lookup_poster_path_by_id.clear()

    assert app.lookup_poster_path_by_id(999, "test-key") is None
    assert app.lookup_poster_path_by_id(999, "test-key") is None
    assert calls == 1


def test_failed_id_lookup_returns_none(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise OSError("network unavailable")

    monkeypatch.setattr(app, "urlopen", fake_urlopen)
    app.lookup_poster_path_by_id.clear()

    assert app.lookup_poster_path_by_id(862, "test-key") is None


def test_title_fallback_and_direct_image_url(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse(
            {
                "results": [
                    {
                        "title": "Toy Story",
                        "release_date": "1995-10-30",
                        "poster_path": "/toy-story.jpg",
                    }
                ]
            }
        )

    monkeypatch.setattr(app, "urlopen", fake_urlopen)
    monkeypatch.setattr(app, "tmdb_api_key", lambda: "test-key")
    app.lookup_poster_path_by_title.clear()
    links = pd.DataFrame([{"movieId": 1, "tmdbId": None}])
    movie = pd.Series(
        {"movieId": 1, "title": "Toy Story (1995)", "release_year": 1995}
    )

    assert app.poster_url(movie, links) == "https://image.tmdb.org/t/p/w500/toy-story.jpg"
    query = parse_qs(urlparse(captured["url"]).query)
    assert query["query"] == ["Toy Story"]
    assert query["year"] == ["1995"]


def test_poster_url_uses_links_tmdb_id_before_title_search(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        app,
        "lookup_poster_path_by_id",
        lambda tmdb_id, api_key: calls.append(("id", tmdb_id)) or "/poster.jpg",
    )
    monkeypatch.setattr(
        app,
        "lookup_poster_path_by_title",
        lambda title, year, api_key: calls.append(("title", title)) or None,
    )
    monkeypatch.setattr(app, "tmdb_api_key", lambda: "test-key")
    movie = pd.Series({"movieId": 1, "title": "Toy Story (1995)"})
    links = pd.DataFrame([{"movieId": 1, "tmdbId": 862}])

    assert app.poster_url(movie, links).endswith("/poster.jpg")
    assert calls == [("id", 862)]


def test_local_poster_fallback_contains_movie_title() -> None:
    fallback = app.poster_placeholder("Movie & Title")
    assert fallback.startswith("data:image/svg+xml")
    assert "Movie%20%26amp%3B%20Title" in fallback


def test_movie_selection_stores_actual_movie_id(monkeypatch) -> None:
    state = {}
    monkeypatch.setattr(app.st, "session_state", state)

    app.select_movie(42)

    assert state["selected_movie_id"] == 42


def test_navigation_change_clears_stale_movie_selection(monkeypatch) -> None:
    state = {"previous_page": "Discover Movies", "selected_movie_id": 42}
    monkeypatch.setattr(app.st, "session_state", state)

    app.update_navigation_state("Similar Movies")

    assert "selected_movie_id" not in state
    assert state["previous_page"] == "Similar Movies"


def test_navigation_rerun_keeps_selection_on_same_page(monkeypatch) -> None:
    state = {"previous_page": "Discover Movies", "selected_movie_id": 42}
    monkeypatch.setattr(app.st, "session_state", state)

    app.update_navigation_state("Discover Movies")

    assert state["selected_movie_id"] == 42
