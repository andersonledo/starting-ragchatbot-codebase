import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# CourseSearchTool tests
# ---------------------------------------------------------------------------

def test_execute_does_not_raise_zero_division(mock_vector_store, sample_search_results):
    """Verifies the ZeroDivisionError bug has been removed from execute()."""
    mock_vector_store.search.return_value = sample_search_results
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

    tool = CourseSearchTool(mock_vector_store)
    # Should not raise ZeroDivisionError after the fix
    result = tool.execute(query="test query")
    assert result is not None


def test_execute_returns_formatted_results(mock_vector_store, sample_search_results):
    """After fix: execute returns a formatted string containing the course title."""
    mock_vector_store.search.return_value = sample_search_results
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="test query")

    assert "[Python Course" in result


def test_execute_no_results_returns_message(mock_vector_store, empty_search_results):
    """After fix: empty results return a 'No relevant content found.' message."""
    mock_vector_store.search.return_value = empty_search_results

    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="test query")

    assert "No relevant content found" in result


def test_execute_tracks_last_sources(mock_vector_store, sample_search_results):
    """After fix: last_sources is populated after a successful search."""
    mock_vector_store.search.return_value = sample_search_results
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

    tool = CourseSearchTool(mock_vector_store)
    tool.execute(query="test query")

    assert len(tool.last_sources) > 0


def test_execute_with_course_filter(mock_vector_store, sample_search_results):
    """After fix: course_name is forwarded to store.search() as a keyword argument."""
    mock_vector_store.search.return_value = sample_search_results

    tool = CourseSearchTool(mock_vector_store)
    tool.execute(query="test query", course_name="Python Course")

    mock_vector_store.search.assert_called_once_with(
        query="test query",
        course_name="Python Course",
        lesson_number=None
    )


# ---------------------------------------------------------------------------
# ToolManager tests
# ---------------------------------------------------------------------------

def test_tool_manager_dispatches_correctly(mock_vector_store, sample_search_results):
    """After fix: ToolManager.execute_tool dispatches to the correct registered tool."""
    mock_vector_store.search.return_value = sample_search_results
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

    manager = ToolManager()
    tool = CourseSearchTool(mock_vector_store)
    manager.register_tool(tool)

    result = manager.execute_tool("search_course_content", query="test query")

    assert "[Python Course" in result


def test_tool_manager_unknown_tool():
    """ToolManager returns an error string for an unknown tool name."""
    manager = ToolManager()
    result = manager.execute_tool("nonexistent_tool", query="test")

    assert "Tool 'nonexistent_tool' not found" in result


def test_get_last_sources_returns_sources(mock_vector_store, sample_search_results):
    """After fix: ToolManager.get_last_sources() returns non-empty list after a search."""
    mock_vector_store.search.return_value = sample_search_results
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

    manager = ToolManager()
    tool = CourseSearchTool(mock_vector_store)
    manager.register_tool(tool)

    manager.execute_tool("search_course_content", query="test query")
    sources = manager.get_last_sources()

    assert len(sources) > 0
