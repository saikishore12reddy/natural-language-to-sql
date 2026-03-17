import json
from typing import Dict, Any, List
import asyncio
import contextlib
import logging
from openai import AsyncOpenAI

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

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a user query by calling the LLM and routing any tool calls 
        to the MCP server.
        """
        if not self._session:
            await self.connect()

        # Get tools from MCP server
        response = await self._session.list_tools()
        
        # Convert MCP tools to OpenAI tool format
        openai_tools = []
        for tool in response.tools:
            # Reconstruct JSON schema params
            properties = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful SQL Assistant with access to a database via tools. "
                    "When a user asks a question, follow this strict procedure: "
                    "1. Call `get_schema` to see the tables. "
                    "2. Based on the schema, call `execute_sql` with a valid SELECT query. "
                    "3. Answer the user based on the results. "
                    "Always use the provided tools. Do not output anything other than tool calls until you have the data."
                )
            },
            {"role": "user", "content": query}
        ]

        # Call OpenAI (Groq)
        response = await self.openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=openai_tools,
            tool_choice="required"
        )
        logger.info(f"Response: {response}")

        message = response.choices[0].message
        messages.append(message)
        logger.info(f"Message: {message}")
        
        tool_results = []
        final_text = ""

        # Process Tool Calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments_raw = tool_call.function.arguments
                if not arguments_raw or arguments_raw == "null":
                    tool_args = {}
                else:
                    tool_args = json.loads(arguments_raw)
                
                raw_result = None
                try:
                    result = await self._session.call_tool(tool_name, arguments=tool_args)
                    result_text = "\n".join([c.text for c in result.content if hasattr(c, 'text')])
                    
                    # We can try to parse the dict out of the result_text for structural API needs,
                    # but since we own the MCP tool, let's just make sure we capture it
                    if tool_name == "execute_sql":
                        try:
                            # The MCP tool returned string format: "Query executed successfully. Row count: X\nResults: [...]"
                            # We'll just extract it as best as possible, or better yet, we should update server.py to return JSON.
                            import ast
                            lines = result_text.split("\n")
                            if len(lines) > 1 and "Results: " in lines[-1]:
                                results_str = lines[-1].split("Results: ", 1)[1]
                                data = ast.literal_eval(results_str)
                                count_str = lines[0].split("Row count: ")[1]
                                raw_result = {"data": data, "row_count": int(count_str.strip())}
                        except Exception:
                            pass
                            
                except Exception as e:
                    result_text = f"Error executing tool {tool_name}: {str(e)}"
                
                tr = {
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": result_text
                }
                if raw_result:
                    tr["raw_result"] = raw_result
                tool_results.append(tr)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text
                })

            # Call OpenAI again with the tool results to get final answer
            final_response = await self.openai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            final_text = final_response.choices[0].message.content
        else:
            final_text = message.content
            
        to_return = {
            "query": query,
            "response": final_text,
            "tool_calls": tool_results
        }
        
        # Try to extract structured SQL and Data from execute_sql tool call
        for tc in tool_results:
            if tc["tool"] == "execute_sql":
                if "query" in tc["arguments"]:
                    to_return["sql_executed"] = tc["arguments"]["query"]
                if "raw_result" in tc:
                    to_return["data"] = tc["raw_result"].get("data", [])
                    to_return["row_count"] = tc["raw_result"].get("row_count", 0)
                    
        return to_return
