import pytest
from unittest.mock import MagicMock, patch

from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rag(mock_config):
    """RAGSystem with all external dependencies patched out.

    After construction, tool_manager is replaced with a full MagicMock so
    individual tests can control its behaviour without triggering real tool
    execution (including the deliberate ZeroDivisionError in CourseSearchTool).
    """
    with patch("rag_system.VectorStore"), \
         patch("rag_system.AIGenerator"), \
         patch("rag_system.SessionManager"), \
         patch("rag_system.DocumentProcessor"):
        system = RAGSystem(mock_config)

    # Replace tool_manager with a controllable mock
    mock_tm = MagicMock()
    mock_tm.get_tool_definitions.return_value = [{"name": "search_course_content"}]
    mock_tm.get_last_sources.return_value = []
    system.tool_manager = mock_tm

    return system


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_query_returns_response_and_sources(rag):
    """query() returns a (str, list) tuple."""
    rag.ai_generator.generate_response.return_value = "Test response"

    result = rag.query("What is Python?")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


def test_query_propagates_to_ai_generator(rag):
    """ai_generator.generate_response is called and the user query is in the prompt."""
    rag.ai_generator.generate_response.return_value = "Answer"

    rag.query("What is machine learning?")

    call_args = rag.ai_generator.generate_response.call_args
    # The prompt wraps the original query; ensure the content is forwarded
    assert call_args is not None
    query_arg = call_args[1].get("query") or call_args[0][0]
    assert "machine learning" in query_arg


def test_query_tools_passed_to_generator(rag):
    """Tool definitions from tool_manager are forwarded into the generator call."""
    tool_defs = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
    rag.tool_manager.get_tool_definitions.return_value = tool_defs
    rag.ai_generator.generate_response.return_value = "Answer"

    rag.query("What is Python?")

    call_kwargs = rag.ai_generator.generate_response.call_args[1]
    assert call_kwargs["tools"] == tool_defs


def test_query_sources_retrieved_and_reset(rag):
    """get_last_sources() and reset_sources() are both called on tool_manager."""
    rag.ai_generator.generate_response.return_value = "Answer"
    rag.tool_manager.get_last_sources.return_value = [
        {"label": "Python Course", "url": "https://example.com"}
    ]

    rag.query("Test query")

    rag.tool_manager.get_last_sources.assert_called_once()
    rag.tool_manager.reset_sources.assert_called_once()


def test_query_session_history_used(rag):
    """If session_id is provided, get_conversation_history is called with it."""
    rag.ai_generator.generate_response.return_value = "Answer"
    rag.session_manager.get_conversation_history.return_value = "Previous conversation"

    rag.query("Test query", session_id="session_123")

    rag.session_manager.get_conversation_history.assert_called_once_with("session_123")


def test_query_exception_raises_correctly(rag):
    """If ai_generator raises, the exception propagates out of query()."""
    rag.ai_generator.generate_response.side_effect = RuntimeError("API error")

    with pytest.raises(RuntimeError, match="API error"):
        rag.query("Test query")
