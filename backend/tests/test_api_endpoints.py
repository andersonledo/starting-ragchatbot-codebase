"""Tests for the FastAPI HTTP endpoints defined in backend/app.py.

The api_client fixture (conftest.py) imports app.py with external deps mocked:
  - RAGSystem is replaced by a configurable MagicMock.
  - StaticFiles is replaced by a lightweight stub so a real frontend/ directory
    is not required.
"""
import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

def test_query_success(api_client):
    """Valid query returns 200 with answer, sources, and session_id."""
    response = api_client.post(
        "/api/query",
        json={"query": "What is Python?", "session_id": "sess-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Python is a high-level programming language."
    assert body["session_id"] == "sess-1"
    assert isinstance(body["sources"], list)


def test_query_generates_session_when_none_provided(api_client, mock_rag_instance):
    """When session_id is omitted, a new session is created and returned."""
    response = api_client.post("/api/query", json={"query": "What is Python?"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "new-session-123"
    mock_rag_instance.session_manager.create_session.assert_called_once()


def test_query_uses_provided_session_id(api_client, mock_rag_instance):
    """Provided session_id is forwarded to rag_system.query and echoed back."""
    response = api_client.post(
        "/api/query",
        json={"query": "What is Python?", "session_id": "existing-session"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "existing-session"
    mock_rag_instance.query.assert_called_once_with(
        "What is Python?", "existing-session"
    )


def test_query_includes_sources(api_client):
    """Sources returned by rag_system.query are present in the response body."""
    response = api_client.post(
        "/api/query",
        json={"query": "Tell me about Python", "session_id": "sess-src"},
    )
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) == 1
    assert sources[0]["label"] == "Python Basics - Lesson 1"
    assert sources[0]["url"] == "https://example.com/lesson1"


def test_query_returns_500_on_rag_error(api_client, mock_rag_instance):
    """When rag_system.query raises an exception, the endpoint returns HTTP 500."""
    mock_rag_instance.query.side_effect = RuntimeError("Something went wrong")
    response = api_client.post(
        "/api/query",
        json={"query": "Bad query", "session_id": "sess-err"},
    )
    assert response.status_code == 500
    assert "Something went wrong" in response.json()["detail"]


def test_query_rejects_missing_query_field(api_client):
    """Omitting the required 'query' field returns HTTP 422 (validation error)."""
    response = api_client.post("/api/query", json={"session_id": "sess-val"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

def test_courses_returns_stats(api_client):
    """GET /api/courses returns total_courses count and course_titles list."""
    response = api_client.get("/api/courses")
    assert response.status_code == 200
    body = response.json()
    assert body["total_courses"] == 2
    assert body["course_titles"] == ["Python Basics", "FastAPI Course"]


def test_courses_returns_empty_list_when_no_courses(api_client, mock_rag_instance):
    """When there are no courses, total_courses is 0 and course_titles is []."""
    mock_rag_instance.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": [],
    }
    response = api_client.get("/api/courses")
    assert response.status_code == 200
    body = response.json()
    assert body["total_courses"] == 0
    assert body["course_titles"] == []


def test_courses_returns_500_on_analytics_error(api_client, mock_rag_instance):
    """When get_course_analytics raises an exception, the endpoint returns HTTP 500."""
    mock_rag_instance.get_course_analytics.side_effect = RuntimeError("DB error")
    response = api_client.get("/api/courses")
    assert response.status_code == 500
    assert "DB error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

def test_clear_session_returns_cleared_status(api_client, mock_rag_instance):
    """DELETE /api/session/{id} clears the session and returns {"status": "cleared"}."""
    response = api_client.delete("/api/session/test-session-id")
    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}
    mock_rag_instance.session_manager.clear_session.assert_called_once_with(
        "test-session-id"
    )


# ---------------------------------------------------------------------------
# GET / (static frontend served by mounted DevStaticFiles)
# ---------------------------------------------------------------------------

def test_root_serves_frontend(api_client):
    """The root path is handled by the mounted static-file app and returns 200."""
    response = api_client.get("/")
    assert response.status_code == 200
