import sys
import json
import os
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

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a user query by delegating to the LangGraph-based orchestration pipeline.
        """
        if not self._session:
            await self.connect()
            
        return await run_nl2sql_graph(query, self)
