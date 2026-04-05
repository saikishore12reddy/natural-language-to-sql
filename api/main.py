from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read API Key from environment
api_key = os.getenv("GROQ_API_KEY")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    type: str  # success, clarification, error
    query: str
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = 0
    message: Optional[str] = None
    rewritten_query: Optional[str] = None
    options: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    missing_slots: Optional[List[str]] = None
    slot_analysis: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    intent_analysis: Optional[Dict[str, Any]] = None

@app.post("/query", response_model=QueryResponse)
async def process_natural_language_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    logger.info(f"Received query: {request.query}")
    
    agent = MCPAgent(api_key=api_key)
    
    try:
        result = await agent.process_query(request.query)
        
        return QueryResponse(
            type=result.get("type", "success"),
            query=request.query,
            sql=result.get("sql_executed"),
            data=result.get("data", []),
            row_count=result.get("row_count", 0),
            message=result.get("response") or result.get("message"),
            rewritten_query=result.get("rewritten_query"),
            options=result.get("options"),
            suggestions=result.get("suggestions"),
            missing_slots=result.get("missing"),
            slot_analysis=result.get("slot_analysis"),
            tool_calls=result.get("tool_calls"),
            intent_analysis=result.get("intent_analysis")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await agent.disconnect()
