import sys
import json
import os
import re
from typing import Dict, Any, List, Optional
import asyncio
import contextlib
import logging
import difflib
from openai import AsyncOpenAI

from agent.slot_extractor import SlotExtractor
from agent.clarification_engine import ClarificationEngine
from agent.nl2sql_graph import run_nl2sql_graph

logger = logging.getLogger(__name__)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# We start the server using the same python interpreter
SERVER_COMMAND = [sys.executable, "-m", "mcp_server.server"]

SCHEMA_FILTER_ENABLED = os.getenv("SCHEMA_FILTER_ENABLED", "true").lower() != "false"
MAX_SCHEMA_CANDIDATES = int(os.getenv("MAX_SCHEMA_CANDIDATES", "8"))

class MCPAgent:
    def __init__(self, api_key: str = None):
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            # Important: pass the current environment to ensure python paths
            env=None
        )
        self._session = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._slot_extractor = SlotExtractor(self.openai_client)
        self._clarification_engine = ClarificationEngine(self.openai_client)

        # Session store: session_id -> list of conversation turns
        # Each turn: {"role": "user"|"assistant", "content": "...", "tables": [...]}
        self.session_store: Dict[str, List[Dict]] = {}

    def _get_session_history(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history for a session."""
        return self.session_store.get(session_id, [])

    def _save_session_history(self, session_id: str, history: List[Dict]) -> None:
        """Save conversation history for a session."""
        self.session_store[session_id] = history[-50:]  # Keep last 50 turns to bound memory

    def should_inject_history(self, confidence: float, query: str, session_id: Optional[str]) -> bool:
        """
        Determine if conversation history should be injected into the prompt.

        Returns True when:
        1. Confidence is low (< 0.65) - query is ambiguous
        2. Query contains referential language ("it", "them", "same", etc.)
        3. It's a follow-up query (session exists and query is substantive)
        """
        if not session_id:
            return False

        # Low confidence - need context to disambiguate
        if confidence < 0.65:
            return True

        # Referential language indicating need for context
        referential_pattern = r"\b(?:it|they|them|same|previous|above|those|these|that|those|then)\b"
        if re.search(referential_pattern, query, re.IGNORECASE):
            return True

        # Follow-up query (substantive query with existing session)
        history = self._get_session_history(session_id)
        if history and len(query.strip().split()) > 3:  # More than just a few words
            return True

        return False

    def _build_history_context(self, history: List[Dict[str, str]], max_turns: int = 3) -> str:
        """
        Build a concise context string from conversation history.
        Only includes the most recent turns to avoid overwhelming the LLM.
        """
        if not history:
            return ""

        # Take the most recent turns
        recent_turns = history[-max_turns:] if len(history) > max_turns else history

        # Format as a readable context
        context_parts = []
        for turn in recent_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if content:
                context_parts.append(f"{role}: {content}")

        return " | ".join(context_parts)

        # Session store: session_id -> list of conversation turns
        # Each turn: {"role": "user"|"assistant", "content": "..."}
        self.session_store: Dict[str, List[Dict[str, str]]] = {}
        
    async def connect(self):
        """Connects to the local MCP server over stdio."""
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    async def disconnect(self):
        """Disconnects the MCP client."""
        await self._exit_stack.aclose()

    async def analyze_intent(self, query: str, table_names: List[str]) -> Dict[str, Any]:
        """
        Analyzes the user query to detect intent, confidence, and entities.
        Includes relevant table names in context (filtered by query if enabled) to improve scope detection.
        """
        system_prompt = (
            "Analyze the following user query for an NL2SQL system. "
            f"Available tables: {', '.join(table_names)}. "
            "Return a JSON object with: "
            "1. 'intent': A short description of what the user wants. "
            "2. 'confidence': A score between 0.0 and 1.0. If the query is out of scope (not about the available tables), the confidence should be < 0.3. "
            "3. 'entities': A dictionary of recognized entities (tables, columns, values). "
            "4. 'is_multi_intent': Boolean, true if multiple independent actions are requested. "
            "5. 'needs_clarification': Boolean, true if the query is ambiguous, mapping to multiple tables, or missing filters. "
            "6. 'rewritten_query': A cleaner, more structured version of the user's natural language query that explicitly mentions relevant tables and expected filters. "
            "Return ONLY the JSON object."
        )
        
        response = await self.openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error parsing intent analysis: {e}")
            return {"intent": "unknown", "confidence": 0.0, "entities": {}, "is_multi_intent": False, "needs_clarification": True, "rewritten_query": query}

    async def get_table_names(self, query: Optional[str] = None) -> List[str]:
        """
        Fetches table names from the MCP server.

        Args:
            query: Optional. If provided and SCHEMA_FILTER_ENABLED, returns filtered table names.
        """
        schema_dict = await self.get_schema_dict(query)
        return list(schema_dict.keys())

    async def get_schema_dict(self, query: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Returns the database schema as {table_name: [col1, col2, ...]}.
        Used by SlotExtractor and ClarificationEngine for schema-aware processing.

        Args:
            query: Optional. If provided and SCHEMA_FILTER_ENABLED, uses filtered schema.
        """
        if not self._session:
            await self.connect()

        # Decide whether to use filtering
        if query and SCHEMA_FILTER_ENABLED:
            # Use filtered schema MCP tool
            schema_result = await self._session.call_tool(
                "get_filtered_schema",
                arguments={"query": query, "max_candidates": MAX_SCHEMA_CANDIDATES}
            )
            tool_used = "get_filtered_schema"
        else:
            # Fallback to full schema
            schema_result = await self._session.call_tool("get_schema", arguments={})
            tool_used = "get_schema"

        schema_text = "".join([c.text for c in schema_result.content if hasattr(c, 'text')])

        logger.debug("Fetched schema using %s (query=%r)", tool_used, query or "")

        schema_dict: Dict[str, List[str]] = {}
        current_table: Optional[str] = None
        for line in schema_text.split("\n"):
            if line.startswith("Table: "):
                current_table = line.split("Table: ")[1].strip()
                schema_dict[current_table] = []
            elif line.strip().startswith("Columns: ") and current_table:
                cols_str = line.strip().removeprefix("Columns: ")
                # Each column looks like "name (TYPE)", extract the name part
                cols = [c.split("(")[0].strip() for c in cols_str.split(",") if c.strip()]
                schema_dict[current_table] = cols
        return schema_dict

    async def fuzzy_match_schema(self, query: str, table_names: List[str]) -> List[str]:
        """
        Checks for unknown entities in the query and suggests corrections from the schema.
        """
        suggestions = []
        words = query.lower().split()
        for word in words:
            # Look for close matches to table names
            matches = difflib.get_close_matches(word, table_names, n=1, cutoff=0.6)
            if matches and matches[0] != word:
                suggestions.append(matches[0])
                
        return sorted(list(set(suggestions)))

    async def process_query(self, query: str,
                         conversation_history: Optional[List[Dict[str, str]]] = None,
                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a user query by delegating to the LangGraph-based orchestration pipeline.

        Args:
            query: The natural language query.
            conversation_history: Optional list of previous conversation turns.
            session_id: Optional ID for the conversation session.
        """
        if not self._session:
            await self.connect()

        # Determine effective session_id and history
        effective_session_id = session_id

        # Merge incoming history with stored history
        effective_history = []
        if conversation_history:
            effective_history = conversation_history

        if effective_session_id:
            stored = self._get_session_history(effective_session_id)
            # Prefer incoming history if provided, otherwise use stored
            if not effective_history and stored:
                effective_history = stored
            # Add current user query to history (but don't save until we have a response)
            effective_history.append({"role": "user", "content": query})

        result = await run_nl2sql_graph(
            query, self,
            conversation_history=effective_history,
            session_id=effective_session_id
        )

        # Save updated history if we had a session
        if effective_session_id and effective_history:
            # Extract table names from the result
            tables_used = []
            if result.get("sql_executed"):
                match = re.search(r'\bfrom\s+(\w+)', result["sql_executed"], re.IGNORECASE)
                if match:
                    tables_used.append(match.group(1).lower())
            elif result.get("table_names"):
                tables_used = result.get("table_names")

            # Extract assistant response from result
            assistant_content = result.get("response") or result.get("message") or ""
            if result.get("type") == "success" and result.get("sql_executed"):
                assistant_content += f"\nSQL: {result['sql_executed']}"

            final_history = effective_history
            if assistant_content:
                final_history.append({"role": "assistant", "content": assistant_content, "tables": tables_used})

            self._save_session_history(effective_session_id, final_history)

        # Attach session_id and updated history to result
        if effective_session_id:
            result["session_id"] = effective_session_id
            result["conversation_history"] = self._get_session_history(effective_session_id)

        return result
