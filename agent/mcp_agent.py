import json
from typing import Dict, Any, List, Optional
import asyncio
import contextlib
import logging
import difflib
from openai import AsyncOpenAI

from agent.slot_extractor import SlotExtractor
from agent.clarification_engine import ClarificationEngine

logger = logging.getLogger(__name__)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# We start the server using the virtual environment python
# to ensure it has all dependencies correctly resolved
SERVER_COMMAND = ["python", "-m", "mcp_server.server"]

class MCPAgent:
    def __init__(self, api_key: str = None):
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.server_params = StdioServerParameters(
            command="python",
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
        Includes table names in context to improve scope detection.
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

    async def get_table_names(self) -> List[str]:
        """Fetches table names from the MCP server."""
        if not self._session:
            await self.connect()
        schema_result = await self._session.call_tool("get_schema", arguments={})
        schema_text = "".join([c.text for c in schema_result.content if hasattr(c, 'text')])
        return [line.split("Table: ")[1].strip() for line in schema_text.split("\n") if line.startswith("Table: ")]

    async def get_schema_dict(self) -> Dict[str, List[str]]:
        """
        Returns the database schema as {table_name: [col1, col2, ...]}.
        Used by SlotExtractor and ClarificationEngine for schema-aware processing.
        """
        if not self._session:
            await self.connect()
        schema_result = await self._session.call_tool("get_schema", arguments={})
        schema_text = "".join([c.text for c in schema_result.content if hasattr(c, 'text')])

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
        Processes a user query through the full pipeline:
          Phase 0 — Schema fetch
          Phase 1 — Slot extraction & incomplete query detection  [NEW]
          Phase 2 — Intent & confidence analysis
          Phase 3 — SQL generation via MCP tools
        """
        if not self._session:
            await self.connect()

        # Phase 0: Fetch current schema context (tables + columns)
        schema_dict = await self.get_schema_dict()
        table_names = list(schema_dict.keys())

        # ------------------------------------------------------------------
        # Phase 1: Slot Extraction — catch incomplete queries BEFORE intent
        # ------------------------------------------------------------------
        slot_result = await self._slot_extractor.extract(query, schema_dict)
        if slot_result.is_incomplete():
            return await self._clarification_engine.generate(query, slot_result, schema_dict)

        # Phase 2: Intent & Confidence Analysis
        intent_analysis = await self.analyze_intent(query, table_names)
        confidence = intent_analysis.get("confidence", 0.0)
        rewritten_query = intent_analysis.get("rewritten_query", query)
        
        # Threshold logic
        if intent_analysis.get("is_multi_intent"):
            return {
                "type": "clarification",
                "message": "Your query contains multiple actions. Please choose one to proceed.",
                "rewritten_query": rewritten_query,
                "options": ["Complete first action", "Complete second action"], # Simplified for demo
                "intent_analysis": intent_analysis
            }
            
        if confidence < 0.4:
            suggestions = await self.fuzzy_match_schema(query, table_names)
            return {
                "type": "error",
                "message": "I'm not sure I understand. Could you please rephrase or be more specific?",
                "rewritten_query": rewritten_query,
                "suggestions": suggestions,
                "intent_analysis": intent_analysis
            }
            
        if confidence < 0.75 or intent_analysis.get("needs_clarification"):
            # Try to generate options based on intent
            clarification_prompt = (
                f"The user said: '{query}'. This is a bit ambiguous. "
                "Suggest 2-3 specific ways they might want to filter or refine this query. "
                "Each option should be a full, complete natural language query that the system can process directly. "
                "Return a JSON object with 'message' and 'options' (list of strings)."
            )
            clarification_res = await self.openai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": clarification_prompt}],
                response_format={"type": "json_object"}
            )
            clarification_data = json.loads(clarification_res.choices[0].message.content)
            return {
                "type": "clarification",
                "message": clarification_data.get("message", "Could you clarify your request?"),
                "rewritten_query": rewritten_query,
                "options": clarification_data.get("options", []),
                "intent_analysis": intent_analysis
            }

        # Phase 2: SQL Generation (Proceed with high confidence)
        # Get tools from MCP server
        response = await self._session.list_tools()
        openai_tools = []
        for tool in response.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful SQL Assistant with access to a database via tools. "
                    "Always use the provided tools. If SQL generation fails or you lack context, "
                    "explain why and suggest filters. Do NOT guess schema or business logic."
                )
            },
            {"role": "user", "content": f"Query: {query}\nIntent Analysis: {json.dumps(intent_analysis)}"}
        ]

        response = await self.openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto" 
        )

        message = response.choices[0].message
        messages.append(message)
        
        tool_results = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                
                try:
                    result = await self._session.call_tool(tool_name, arguments=tool_args)
                    result_text = "\n".join([c.text for c in result.content if hasattr(c, 'text')])
                    
                    raw_result = None
                    if tool_name == "execute_sql":
                        try:
                            import ast
                            lines = result_text.split("\n")
                            if len(lines) > 1 and "Results: " in lines[-1]:
                                results_str = lines[-1].split("Results: ", 1)[1]
                                data = ast.literal_eval(results_str)
                                count_str = lines[0].split("Row count: ")[1]
                                raw_result = {"data": data, "row_count": int(count_str.strip())}
                        except: pass
                            
                except Exception as e:
                    result_text = f"Error: {str(e)}"
                
                tool_results.append({"tool": tool_name, "arguments": tool_args, "result": result_text, "raw_result": raw_result})
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})

            final_response = await self.openai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            final_text = final_response.choices[0].message.content
        else:
            final_text = message.content or "I couldn't generate a valid action for this query."

        to_return = {
            "type": "success",
            "query": query,
            "rewritten_query": rewritten_query,
            "response": final_text,
            "tool_calls": tool_results
        }
        
        for tc in tool_results:
            if tc["tool"] == "execute_sql":
                to_return["sql_executed"] = tc["arguments"].get("query")
                if tc.get("raw_result"):
                    to_return["data"] = tc["raw_result"].get("data", [])
                    to_return["row_count"] = tc["raw_result"].get("row_count", 0)
                    
        return to_return
