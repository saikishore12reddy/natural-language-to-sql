from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from agent.mcp_agent import MCPAgent
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI-Powered NL to SQL System", version="1.0.0")

# Read API Key from environment
api_key = os.getenv("GROQ_API_KEY")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = 0
    message: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

@app.post("/query", response_model=QueryResponse)
async def process_natural_language_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    logger.info(f"Received query: {request.query}")
    
    # Instantiate the agent. In a production app, we might pool or persist the agent connection.
    # We create and connect per request here for stateless simplicity.
    agent = MCPAgent(api_key=api_key)
    
    try:
        result = await agent.process_query(request.query)
        
        # We try to extract structured SQL and Data from the tool execution
        # if the 'execute_sql' tool was called.
        sql_executed = None
        data = []
        row_count = 0
        
        for tc in result.get("tool_calls", []):
            if tc.get("tool") == "execute_sql":
                sql_executed = tc.get("arguments", {}).get("query")
                
                # The text result from the MCP tool has the structured dictionary format embedded in it.
                # In mcp_server we returned: "Query executed successfully. Row count: X\nResults: [{...}]"
                # Since we don't return pure JSON from the python MCP SDK execute method right now,
                # we pass back the raw agent processing in `result` and the conversational response.
                # To purely conform to the PRD JSON structure:
                pass
                
        # Re-evaluating the extraction logic: The LLM agent returns a JSON with 'response', 'tool_calls'
        # The tool_calls have `arguments` (which has the query).
        # We can fetch the raw data by parsing the tool_call results or we can adjust `mcp_agent` to return it securely.
        
        return QueryResponse(
            query=result.get("query"),
            sql=result.get("sql_executed"),
            data=result.get("data", []),
            row_count=result.get("row_count", 0),
            message=result.get("response"),
            tool_calls=result.get("tool_calls")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await agent.disconnect()
