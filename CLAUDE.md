# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
uv sync
```

**Run the server:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app serves at `http://localhost:8000` and the API docs at `http://localhost:8000/docs`.

**Environment setup:** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

This is a full-stack RAG chatbot. The FastAPI backend (`backend/`) serves both the API and the static frontend (`frontend/`).

### Request flow

1. User submits a query via the web UI → `POST /api/query`
2. `app.py` calls `RAGSystem.query()` — the central orchestrator
3. `RAGSystem` sends the query to Claude (`AIGenerator`) along with a `search_course_content` tool definition
4. Claude decides whether to invoke the tool; if so, `ToolManager` dispatches to `CourseSearchTool`
5. `CourseSearchTool` calls `VectorStore.search()`, which does semantic search via ChromaDB
6. Tool results are fed back to Claude for a final answer
7. Sources (course/lesson citations) are tracked by `CourseSearchTool.last_sources` and returned alongside the answer

### Key components

| File | Role |
|---|---|
| `backend/rag_system.py` | Orchestrator — wires all components together |
| `backend/ai_generator.py` | Anthropic Claude API calls; handles tool-use loop |
| `backend/vector_store.py` | ChromaDB wrapper; two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `backend/document_processor.py` | Parses `.txt`/`.pdf`/`.docx` course files into `Course`/`Lesson`/`CourseChunk` objects |
| `backend/search_tools.py` | `Tool` ABC, `CourseSearchTool`, and `ToolManager` registry |
| `backend/session_manager.py` | In-memory conversation history keyed by session ID |
| `backend/config.py` | Single `Config` dataclass; all tuneable parameters live here |
| `backend/models.py` | Pydantic/dataclass models: `Course`, `Lesson`, `CourseChunk` |

### Course document format

Course files in `docs/` must follow this structure for `DocumentProcessor` to parse them correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

Documents are auto-ingested from `docs/` on server startup. Already-indexed courses (matched by title) are skipped.

### ChromaDB storage

ChromaDB persists to `backend/chroma_db/` (created at runtime, gitignored). Two collections:
- `course_catalog` — one document per course, stores title/instructor/link/lesson index as metadata
- `course_content` — chunked lesson text; metadata includes `course_title`, `lesson_number`, `chunk_index`

Course title is used as the ChromaDB document ID, so titles must be unique.

### Configuration knobs (`backend/config.py`)

- `ANTHROPIC_MODEL` — Claude model used for generation (default: `claude-sonnet-4-20250514`)
- `EMBEDDING_MODEL` — SentenceTransformer model for embeddings (default: `all-MiniLM-L6-v2`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — text chunking parameters (800 / 100 chars)
- `MAX_RESULTS` — number of chunks returned per search (default: 5)
- `MAX_HISTORY` — conversation turns retained per session (default: 2)
- `CHROMA_PATH` — ChromaDB persistence path (default: `./chroma_db`, relative to `backend/`)

### Adding new tools

Subclass `Tool` in `backend/search_tools.py`, implement `get_tool_definition()` (returns Anthropic tool schema) and `execute(**kwargs)`, then register with `ToolManager.register_tool()` in `RAGSystem.__init__`.
