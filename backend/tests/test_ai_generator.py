import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------

def make_text_response(text="Direct answer", stop_reason="end_turn"):
    """Build a mock Anthropic response with a text content block."""
    response = MagicMock()
    response.stop_reason = stop_reason
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response.content = [text_block]
    return response


def make_tool_use_response(
    tool_name="search_course_content",
    tool_input=None,
    tool_id="tool_123"
):
    """Build a mock Anthropic response requesting a tool call."""
    if tool_input is None:
        tool_input = {"query": "test query"}

    response = MagicMock()
    response.stop_reason = "tool_use"

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_id
    response.content = [tool_block]
    return response


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ai_generator():
    """AIGenerator instance with a mocked Anthropic client."""
    with patch("anthropic.Anthropic"):
        from ai_generator import AIGenerator
        gen = AIGenerator(api_key="test_key", model="test-model")
    # gen.client is a MagicMock set during __init__; the patch is no longer
    # needed after construction.
    return gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_direct_response_no_tools(ai_generator):
    """When stop_reason == 'end_turn', generate_response returns content[0].text."""
    ai_generator.client.messages.create.return_value = make_text_response("Direct answer")

    result = ai_generator.generate_response("What is Python?")

    assert result == "Direct answer"


def test_tool_use_triggers_handle_tool_execution(ai_generator):
    """When stop_reason == 'tool_use' and tool_manager is supplied, a second API
    call is made and its text is returned."""
    first = make_tool_use_response()
    second = make_text_response("Final answer after tool use")
    ai_generator.client.messages.create.side_effect = [first, second]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "search results"

    result = ai_generator.generate_response(
        "Search for Python", tool_manager=mock_tool_manager
    )

    assert result == "Final answer after tool use"
    assert ai_generator.client.messages.create.call_count == 2


def test_handle_tool_execution_calls_tool_manager(ai_generator):
    """tool_manager.execute_tool is called with the correct tool name and inputs."""
    first = make_tool_use_response(
        tool_name="search_course_content",
        tool_input={"query": "Python basics"},
        tool_id="tool_abc"
    )
    second = make_text_response("Final answer")
    ai_generator.client.messages.create.side_effect = [first, second]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "search result"

    ai_generator.generate_response("What is Python?", tool_manager=mock_tool_manager)

    mock_tool_manager.execute_tool.assert_called_once_with(
        "search_course_content",
        query="Python basics"
    )


def test_handle_tool_execution_sends_tool_result_back(ai_generator):
    """The second API call includes a tool_result message block."""
    first = make_tool_use_response(tool_id="tool_xyz")
    second = make_text_response("Final answer")
    ai_generator.client.messages.create.side_effect = [first, second]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "search result content"

    ai_generator.generate_response("Search query", tool_manager=mock_tool_manager)

    assert ai_generator.client.messages.create.call_count == 2
    second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]

    # Find the tool_result content block in the second call's messages
    tool_result_block = None
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_block = block

    assert tool_result_block is not None, "No tool_result block found in second API call"
    assert tool_result_block["tool_use_id"] == "tool_xyz"
    assert tool_result_block["content"] == "search result content"


def test_no_tool_manager_skips_execution(ai_generator):
    """When tool_manager is None, only one API call is made even if stop_reason is
    'tool_use'."""
    ai_generator.client.messages.create.return_value = make_tool_use_response()

    ai_generator.generate_response("Search query", tool_manager=None)

    assert ai_generator.client.messages.create.call_count == 1


def test_two_sequential_tool_rounds(ai_generator):
    """Two sequential tool_use responses followed by a text response require 3 API
    calls and 2 tool executions."""
    first = make_tool_use_response(tool_name="get_course_outline", tool_id="t1")
    second = make_tool_use_response(tool_name="search_course_content", tool_id="t2")
    third = make_text_response("Final combined answer")
    ai_generator.client.messages.create.side_effect = [first, second, third]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "some results"

    result = ai_generator.generate_response(
        "Multi-step query", tool_manager=mock_tool_manager
    )

    assert result == "Final combined answer"
    assert ai_generator.client.messages.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2


def test_intermediate_api_call_includes_tools(ai_generator):
    """When tools are provided and the loop has not reached the last round, the
    intermediate API call includes the tools parameter."""
    first = make_tool_use_response(tool_id="t1")
    second = make_text_response("Done")
    ai_generator.client.messages.create.side_effect = [first, second]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "results"

    fake_tools = [{"name": "search_course_content", "description": "search"}]
    ai_generator.generate_response(
        "query", tools=fake_tools, tool_manager=mock_tool_manager
    )

    # The second call (index 1) is the intermediate call and should include tools
    second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
    assert "tools" in second_call_kwargs


def test_final_round_api_call_excludes_tools(ai_generator):
    """When the loop reaches the last round (round_num == MAX_TOOL_ROUNDS - 1), the
    API call must NOT include tools."""
    first = make_tool_use_response(tool_id="t1")
    second = make_tool_use_response(tool_id="t2")
    third = make_text_response("Done")
    ai_generator.client.messages.create.side_effect = [first, second, third]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "results"

    fake_tools = [{"name": "search_course_content", "description": "search"}]
    ai_generator.generate_response(
        "query", tools=fake_tools, tool_manager=mock_tool_manager
    )

    # The third call (index 2) is the last-round call and must NOT include tools
    third_call_kwargs = ai_generator.client.messages.create.call_args_list[2][1]
    assert "tools" not in third_call_kwargs


def test_tool_failure_terminates_loop(ai_generator):
    """When a tool is not found, the loop exits early and a fallback API call is
    made without tools, returning the fallback text."""
    unknown_tool_response = make_tool_use_response(tool_name="nonexistent", tool_id="t1")
    fallback_text = make_text_response("Fallback answer")
    ai_generator.client.messages.create.side_effect = [unknown_tool_response, fallback_text]

    mock_tool_manager = MagicMock()
    mock_tool_manager.execute_tool.return_value = "Tool 'nonexistent' not found"

    result = ai_generator.generate_response(
        "query", tool_manager=mock_tool_manager
    )

    assert result == "Fallback answer"
    assert ai_generator.client.messages.create.call_count == 2
