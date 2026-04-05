---
description: Track implemented features and edge cases
paths: ["src/**/*"]
---

# Implemented Features

## FastAPI Query Endpoint
**File:** `api/main.py`
**Date:** 2026-04-04
**Description:** Main REST API endpoint (`POST /query`) that accepts natural language queries and returns structured responses including SQL, data, and clarification options.

**Key Details:**
- Accepts `QueryRequest` with query string
- Returns `QueryResponse` with multiple fields: type (success/clarification/error), SQL, data, row counts, messages, options, slot analysis, intent analysis
- Configures CORS middleware for cross-origin requests
- Creates new `MCPAgent` instance per request and disconnects after processing

**Edge Cases:**
- Empty/whitespace-only queries return 400 error
- Agent exceptions return 500 error with details
- Agent is always disconnected in `finally` block to prevent resource leaks
- Groq API key loaded from environment variable (can be None)

**Why it matters:** This is the single entry point for all NL2SQL operations. When adding new API endpoints or modifying response structures, this file must be updated.

---

## Query Clarification Client
**File:** `query_clarifier.py`
**Date:** 2026-04-04
**Description:** Interactive CLI client that sends queries to the NL2SQL API and handles the clarification loop. When the API needs more details, it presents multi-choice options and iteratively refines the query until it can generate SQL.

**Key Details:**
- Uses `aiohttp` for async communication with API
- Runs a `while True` loop: sends query, receives response, if clarification needed → shows options → gets user input → refines query → repeats
- Displays up to 10 result rows in terminal-friendly format
- Validates user input for option selection (1-N or 'q' to quit)

**Edge Cases:**
- API connection failures show helpful message about running the server
- KeyboardInterrupt exits gracefully (Ctrl+C)
- Empty additional details are handled by just using the selected option
- Invalid choice numbers show re-prompt with valid range
- Clarification without options shows "Unexpected clarification" message
- Result type "cancelled" when user quits during clarification

**Why it matters:** This is the user-facing interface. Any changes to API response format or clarification flow require updates here.

---

## MCP Agent with Groq LLM
**File:** `agent/mcp_agent.py`
**Date:** 2026-04-04
**Description:** Core agent that orchestrates NL2SQL processing. Connects to local MCP server via stdio, performs intent analysis using Groq's Llama 3.3 70B model, and delegates to LangGraph pipeline.

**Key Details:**
- Uses `AsyncOpenAI` client pointed at Groq API (`api.groq.com/openai/v1`)
- MCP server runs as subprocess (`python -m mcp_server.server`) via stdio
- Supports schema filtering with `SCHEMA_FILTER_ENABLED` env var (limits to `MAX_SCHEMA_CANDIDATES` tables)
- Fuzzy matches query terms to table names using `difflib`
- `process_query` delegates to `run_nl2sql_graph` (LangGraph workflow)

**Intent Analysis:**
- Returns JSON with: intent, confidence (0-1), entities, is_multi_intent, needs_clarification, rewritten_query
- Queries not about available tables get confidence < 0.3
- Parses Groq JSON response with error fallbacks

**Edge Cases:**
- Intent analysis JSON parsing failures return default dict with `needs_clarification: True`
- Session auto-connects if not already connected
- Session cleanup via `AsyncExitStack`
- Schema parsing handles `get_filtered_schema` vs `get_schema` tool selection
- Column type extraction strips everything after `(` in `"name (TYPE)"` format

**Why it matters:** This is the brain of the system. All query processing flows through here. Adding new LLM features or MCP tools requires changes to this file.

---

## LangGraph NL2SQL Workflow
**File:** `agent/nl2sql_graph.py`
**Date:** 2026-04-04
**Description:** LangGraph-based state machine that orchestrates the complete NL2SQL pipeline. Nodes include: `validate_query` (early check for table references), `fetch_schema`, `extract_slots`, `analyze_intent`, `clarify` (for missing slots), `fallback` (for low confidence/multi-intent), and `generate_sql`. The workflow now includes early validation to avoid unnecessary schema fetches for vague queries, and each node logs its execution for tracing.

**Key Details:**
- Graph state (`NL2SQLState`) tracks query, schema, slots, intent, response, messages, tool results
- Nodes: `validate_query`, `fetch_schema`, `extract_slots`, `analyze_intent`, `clarify`, `fallback`, `generate_sql`
- `validate_query` runs first; if query lacks table references, returns early clarification without DB access
- Conditional routing: after validate → either END or fetch_schema; after slots → clarify if incomplete else analyze_intent; after intent → fallback if low confidence/multi-intent else generate_sql
- All nodes include structured logging (INFO level) for observability

**Edge Cases:**
- `validate_query`: handles empty after stop-words, no table matches, fuzzy matching for typos
- `fetch_schema`: filtered vs full schema fallback
- `extract_slots`: may return incomplete slots triggering clarification
- `analyze_intent`: catches parsing errors, returns needs_clarification=True
- `fallback`: multi-intent detection, confidence-based branching, optional schema suggestions via `fuzzy_match_schema`
- `generate_sql`: tool call handling, result parsing, error capture

**Why it matters:** This is the orchestration core. Changes to workflow order, new processing stages, or modified branching logic happen here. The recent validation addition optimizes performance and reduces unnecessary DB/API calls.

---

## Slot Extractor
**File:** `agent/slot_extractor.py`
**Date:** 12:20 AM, March 22nd, 2026 at 11:41.
**Description:** Extracts structured slots (table references, column names, filter values, aggregation functions) from natural language queries using the LLM.

**Edge Cases:**
- Queries with no recognizable slots return empty slot dict
- Ambiguous column/table mappings flagged for clarification
- Partial slot extraction (some slots found, others missing)

**Why it matters:** Accurate slot extraction is critical for downstream SQL generation. Improvements to extraction logic directly impact query success rate.

---

## Clarification Engine
**File:** `agent/clarification_engine.py`
**Date:** 12:20 AM, March 22nd, 2026 at 11:41.
**Description:** Generates clarification prompts and options when queries are ambiguous or missing required slots.

**Edge Cases:**
- Multiple missing slots prioritized by importance
- Generated options are contextually relevant to available schema
- Handles cases where clarification itself is ambiguous

**Why it matters:** This determines the quality of user-facing clarification prompts. Good clarification reduces user frustration and improves SQL accuracy.

---

## Schema Inspector
**File:** `core/schema_inspector.py`
**Date:** 2026-04-04
**Description:** Database schema inspection module that fetches table and column metadata from the connected database.

**Edge Cases:**
- Database connection failures
- Empty schemas
- Schema changes between inspection and query execution

**Why it matters:** Schema accuracy is foundational to the entire NL2SQL system.

---

## MCP Server
**File:** `mcp_server/server.py`
**Date:** 2026-04-04
**Description:** Local MCP server that exposes database schema and query execution as MCP tools. Connects to the database and provides `get_schema`, `get_filtered_schema`, and query execution tools.

**Edge Cases:**
- Database connectivity issues
- Slow query timeouts
- Authentication failures

**Why it matters:** This is the bridge between the agent and the database.
