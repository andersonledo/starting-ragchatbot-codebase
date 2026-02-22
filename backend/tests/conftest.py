import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add backend directory to path so test files can import backend modules directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


@pytest.fixture
def mock_config():
    """Minimal config mock shared across all test modules."""
    config = MagicMock()
    config.ANTHROPIC_API_KEY = "test_key"
    config.ANTHROPIC_MODEL = "test-model"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma"
    return config


@pytest.fixture
def sample_sources():
    """Sample source data in the format returned by RAGSystem.query()."""
    return [
        {"label": "Python Course · Lesson 1", "url": "https://example.com/lesson1"},
        {"label": "Python Course · Lesson 2", "url": None},
    ]


@pytest.fixture
def sample_search_results():
    """A valid SearchResults object with one document."""
    return SearchResults(
        documents=["This is lesson content about Python basics and data types."],
        metadata=[{"course_title": "Python Course", "lesson_number": 1}],
        distances=[0.1]
    )


@pytest.fixture
def empty_search_results():
    """A SearchResults with empty lists (no results found)."""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def mock_vector_store(sample_search_results):
    """MagicMock of VectorStore with sensible defaults pre-configured."""
    mock = MagicMock()
    mock.search.return_value = sample_search_results
    mock.get_lesson_link.return_value = "https://example.com/lesson1"
    mock.get_course_link.return_value = "https://example.com/course"
    mock.get_course_outline.return_value = {
        "title": "Python Course",
        "course_link": "https://example.com/course",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction to Python"}
        ]
    }
    return mock


# ---------------------------------------------------------------------------
# API-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_instance():
    """Pre-configured MagicMock for RAGSystem, suitable for API-level tests."""
    mock = MagicMock()
    mock.query.return_value = (
        "Python is a high-level programming language.",
        [{"label": "Python Basics - Lesson 1", "url": "https://example.com/lesson1"}],
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "FastAPI Course"],
    }
    mock.session_manager.create_session.return_value = "new-session-123"
    return mock


@pytest.fixture
def api_client(mock_rag_instance):
    """TestClient for the real FastAPI app with all external deps patched.

    A lightweight _MockStaticFiles replaces Starlette's StaticFiles so that
    importing app.py succeeds without a real frontend/ directory on disk.
    The module-level rag_system in app.py is replaced with mock_rag_instance
    so endpoint functions use the test mock for every request.
    """
    from fastapi.testclient import TestClient

    class _MockStaticFiles:
        """Minimal ASGI stub mounted at '/' in place of the real StaticFiles."""

        def __init__(self, *args, **kwargs):
            pass

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return
            from starlette.responses import HTMLResponse
            await HTMLResponse("<html><body>Test frontend</body></html>")(scope, receive, send)

    # Remove any cached import so that the patched symbols take effect.
    sys.modules.pop("app", None)

    with patch("fastapi.staticfiles.StaticFiles", _MockStaticFiles), \
         patch("rag_system.RAGSystem", return_value=mock_rag_instance):
        import app as _app_module

    # Replace the module-level variable so endpoint closures resolve to our mock.
    _app_module.rag_system = mock_rag_instance

    yield TestClient(_app_module.app)

    sys.modules.pop("app", None)
