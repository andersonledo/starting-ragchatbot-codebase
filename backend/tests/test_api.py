"""Tests for the FastAPI API endpoints defined in app.py.

Because app.py creates module-level singletons at import time (a live RAGSystem
and a DevStaticFiles mount that checks for the frontend directory), we patch
those dependencies before the import so the module loads cleanly in the test
environment without ChromaDB, sentence-transformer models, or the frontend
directory being present.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Keep path setup in sync with conftest.py; harmless if already added.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Stub for StaticFiles – replaces the real starlette class so that
# DevStaticFiles(directory="../frontend") succeeds without a real directory.
# ---------------------------------------------------------------------------

class _FakeStaticFiles:
    """Minimal ASGI stub; satisfies the inheritance chain without I/O."""

    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, scope, receive, send):
        # Only reached for unmatched paths – not exercised by API tests.
        pass


# ---------------------------------------------------------------------------
# Module-level RAGSystem mock – created once, reset per test via fixture.
# ---------------------------------------------------------------------------

_mock_rag = MagicMock()

# Patch before importing app.py so module-level code picks up the mocks:
#   • RAGSystem(config)  →  returns _mock_rag
#   • DevStaticFiles(directory="../frontend", ...)  →  no directory check
with patch("rag_system.RAGSystem", return_value=_mock_rag), \
     patch("fastapi.staticfiles.StaticFiles", _FakeStaticFiles):
    import app as _app_module  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_rag():
    """Reset and configure the shared RAGSystem mock for test isolation."""
    _mock_rag.reset_mock()
    _mock_rag.session_manager.create_session.return_value = "new-session-id"
    _mock_rag.query.return_value = ("Default answer", [])
    _mock_rag.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": [],
    }
    return _mock_rag


@pytest.fixture()
def client(mock_rag):
    """Synchronous TestClient wrapping the real FastAPI app."""
    with TestClient(_app_module.app, raise_server_exceptions=True) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_successful_query_returns_200(self, client, mock_rag):
        mock_rag.query.return_value = ("Here is the answer", [])
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, client, mock_rag):
        mock_rag.query.return_value = ("Here is the answer", [])
        data = client.post("/api/query", json={"query": "test"}).json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_answer_matches_rag_output(self, client, mock_rag):
        mock_rag.query.return_value = ("Python is a language", [])
        data = client.post("/api/query", json={"query": "What is Python?"}).json()
        assert data["answer"] == "Python is a language"

    def test_sources_included_in_response(self, client, mock_rag):
        sources = [{"label": "Python Course · Lesson 1", "url": "https://example.com/lesson1"}]
        mock_rag.query.return_value = ("Answer", sources)
        data = client.post("/api/query", json={"query": "test"}).json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["label"] == "Python Course · Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/lesson1"

    def test_source_without_url_is_serialized(self, client, mock_rag):
        sources = [{"label": "Unlinkable Lesson", "url": None}]
        mock_rag.query.return_value = ("Answer", sources)
        data = client.post("/api/query", json={"query": "test"}).json()
        assert data["sources"][0]["url"] is None

    def test_new_session_created_when_none_provided(self, client, mock_rag):
        mock_rag.session_manager.create_session.return_value = "generated-session"
        mock_rag.query.return_value = ("Answer", [])
        data = client.post("/api/query", json={"query": "test"}).json()
        assert data["session_id"] == "generated-session"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_provided_session_id_is_preserved(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        data = client.post(
            "/api/query",
            json={"query": "test", "session_id": "existing-session"},
        ).json()
        assert data["session_id"] == "existing-session"
        mock_rag.session_manager.create_session.assert_not_called()

    def test_rag_query_called_with_correct_args(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        client.post(
            "/api/query",
            json={"query": "What is machine learning?", "session_id": "sess-1"},
        )
        mock_rag.query.assert_called_once_with("What is machine learning?", "sess-1")

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={"session_id": "sess"})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_rag_exception_returns_500(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("Unexpected backend error")
        resp = client.post("/api/query", json={"query": "test"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_courses_returns_200(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course A", "Course B", "Course C"],
        }
        resp = client.get("/api/courses")
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Intro to Python"],
        }
        data = client.get("/api/courses").json()
        assert "total_courses" in data
        assert "course_titles" in data

    def test_total_courses_matches_analytics(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        data = client.get("/api/courses").json()
        assert data["total_courses"] == 5

    def test_course_titles_match_analytics(self, client, mock_rag):
        titles = ["Python Basics", "Machine Learning 101"]
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": titles,
        }
        data = client.get("/api/courses").json()
        assert data["course_titles"] == titles

    def test_empty_catalog_returns_zero_courses(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        data = client.get("/api/courses").json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_analytics_exception_returns_500(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("DB connection failed")
        resp = client.get("/api/courses")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestSessionEndpoint:

    def test_delete_session_returns_200(self, client):
        resp = client.delete("/api/session/some-session-id")
        assert resp.status_code == 200

    def test_delete_session_returns_cleared_status(self, client):
        data = client.delete("/api/session/some-session-id").json()
        assert data == {"status": "cleared"}

    def test_clear_session_called_with_correct_id(self, client, mock_rag):
        client.delete("/api/session/my-session-123")
        mock_rag.session_manager.clear_session.assert_called_once_with("my-session-123")
