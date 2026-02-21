import sys
import os
import pytest
from unittest.mock import MagicMock

# Add backend directory to path so test files can import backend modules directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


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
