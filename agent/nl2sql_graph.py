import json
import logging
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agent.slot_extractor import SlotResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class NL2SQLState(TypedDict):
    """
    State maintained across the LangGraph nodes.
    """
    query: str
    schema_dict: Dict[str, List[str]]
    table_names: List[str]
    slot_result: Optional[SlotResult]
    intent_analysis: Optional[Dict[str, Any]]
    final_response: Optional[Dict[str, Any]]
    messages: List[Dict[str, str]]
    tool_results: List[Dict[str, Any]]

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def fetch_schema_node(state: NL2SQLState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Fetch dynamic schema context from MCP."""
    agent = config["configurable"]["agent"]
    schema_dict = await agent.get_schema_dict()
    return {
        "schema_dict": schema_dict,
        "table_names": list(schema_dict.keys())
    }

async def extract_slots_node(state: NL2SQLState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Extract slots and detect structural incompleteness."""
    agent = config["configurable"]["agent"]
    slot_result = await agent._slot_extractor.extract(state["query"], state["schema_dict"])
    return {"slot_result": slot_result}

async def clarify_node(state: NL2SQLState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Generate clarification for incomplete queries."""
    agent = config["configurable"]["agent"]
    response = await agent._clarification_engine.generate(
        state["query"], state["slot_result"], state["schema_dict"]
    )
    return {"final_response": response}

async def analyze_intent_node(state: NL2SQLState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Perform deep intent analysis with confidence scoring."""
    agent = config["configurable"]["agent"]
    intent_analysis = await agent.analyze_intent(state["query"], state["table_names"])
    return {"intent_analysis": intent_analysis}

async def fallback_node(state: NL2SQLState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Handle low-confidence or ambiguous queries."""
    agent = config["configurable"]["agent"]
    query = state["query"]
    intent_analysis = state["intent_analysis"]
    
    confidence = intent_analysis.get("confidence", 0.0)
    rewritten_query = intent_analysis.get("rewritten_query", query)
    
    if intent_analysis.get("is_multi_intent"):
        return {
            "final_response": {
                "type": "clarification",
                "message": "Your query contains multiple actions. Please choose one to proceed.",
                "rewritten_query": rewritten_query,
                "options": ["Complete first action", "Complete second action"],
                "intent_analysis": intent_analysis
            }
        }
        
    if confidence < 0.4:
        suggestions = await agent.fuzzy_match_schema(query, state["table_names"])
        response = {
            "type": "error",
            "message": "I'm not sure I understand. Could you please rephrase or be more specific?",
            "rewritten_query": rewritten_query,
            "suggestions": suggestions,
            "intent_analysis": intent_analysis
        }
    else:
        # Confidence between 0.4 and 0.75 or needs_clarification=True
        clarification_prompt = (
            f"The user said: '{query}'. This is a bit ambiguous. "
            "Suggest 2-3 specific ways they might want to filter or refine this query. "
            "Each option should be a full, complete natural language query that the system can process directly. "
            "Return a JSON object with 'message' and 'options' (list of strings)."
        )
        clarification_res = await agent.openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": clarification_prompt}],
            response_format={"type": "json_object"}
        )
        clarification_data = json.loads(clarification_res.choices[0].message.content)
        response = {
            "type": "clarification",
            "message": clarification_data.get("message", "Could you clarify your request?"),
            "rewritten_query": rewritten_query,
            "options": clarification_data.get("options", []),
            "intent_analysis": intent_analysis
        }
    
    return {"final_response": response}

async def generate_sql_node(state: NL2SQLState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Generate SQL using MCP tools (High Confidence Path)."""
    agent = config["configurable"]["agent"]
    query = state["query"]
    intent_analysis = state["intent_analysis"]
    
    # Get tools from MCP server
    response = await agent._session.list_tools()
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

    response = await agent.openai_client.chat.completions.create(
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
                result = await agent._session.call_tool(tool_name, arguments=tool_args)
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

        final_response = await agent.openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        final_text = final_response.choices[0].message.content
    else:
        final_text = message.content or "I couldn't generate a valid action for this query."

    to_return = {
        "type": "success",
        "query": query,
        "rewritten_query": intent_analysis.get("rewritten_query", query),
        "response": final_text,
        "tool_calls": tool_results
    }
    
    for tc in tool_results:
        if tc["tool"] == "execute_sql":
            to_return["sql_executed"] = tc["arguments"].get("query")
            if tc.get("raw_result"):
                to_return["data"] = tc["raw_result"].get("data", [])
                to_return["row_count"] = tc["raw_result"].get("row_count", 0)

    return {"final_response": to_return}

# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------

def route_after_slots(state: NL2SQLState) -> str:
    """Route based on slot completeness."""
    if state["slot_result"].is_incomplete():
        return "clarify"
    return "analyze_intent"

def route_after_intent(state: NL2SQLState) -> str:
    """Route based on intent confidence and clarity."""
    intent = state["intent_analysis"]
    confidence = intent.get("confidence", 0.0)
    
    if intent.get("is_multi_intent"):
        return "fallback" # multi-intent handled as ambiguity
        
    if confidence < 0.75 or intent.get("needs_clarification"):
        return "fallback"
        
    return "generate_sql"

# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def create_nl2sql_graph():
    workflow = StateGraph(NL2SQLState)

    # Add Nodes
    workflow.add_node("fetch_schema", fetch_schema_node)
    workflow.add_node("extract_slots", extract_slots_node)
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("generate_sql", generate_sql_node)

    # Build Edges
    workflow.add_edge(START, "fetch_schema")
    workflow.add_edge("fetch_schema", "extract_slots")
    
    workflow.add_conditional_edges(
        "extract_slots",
        route_after_slots,
        {
            "clarify": "clarify",
            "analyze_intent": "analyze_intent"
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_intent,
        {
            "fallback": "fallback",
            "generate_sql": "generate_sql"
        }
    )
    
    workflow.add_edge("clarify", END)
    workflow.add_edge("fallback", END)
    workflow.add_edge("generate_sql", END)

    return workflow.compile()

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

async def run_nl2sql_graph(query: str, agent: Any) -> Dict[str, Any]:
    """Runs the compiled graph for a given query and agent instance."""
    graph = create_nl2sql_graph()
    
    initial_state = {
        "query": query,
        "schema_dict": {},
        "table_names": [],
        "slot_result": None,
        "intent_analysis": None,
        "final_response": None,
        "messages": [],
        "tool_results": []
    }
    
    final_state = await graph.ainvoke(
        initial_state, 
        config={"configurable": {"agent": agent}}
    )
    
    return final_state["final_response"]
