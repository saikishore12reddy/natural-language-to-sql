import os
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from core.database import execute_safe_sql, seed_table
from core.schema_inspector import get_schema_metadata, get_filtered_schema_metadata

# Initialize FastMCP Server
mcp = FastMCP("NL-to-SQL-Server")

@mcp.tool()
def get_schema() -> str:
    """
    Returns the database schema metadata, including tables, columns, and foreign keys.
    Useful for LLM to understand the dynamic database schema.
    """
    return get_schema_metadata()


@mcp.tool()
def get_filtered_schema(query: str, max_candidates: int = 8) -> str:
    """
    Returns filtered schema metadata for only the tables relevant to the query.
    Includes FK-related tables to preserve join context.
    Useful for reducing LLM context when working with large databases.
    """
    return get_filtered_schema_metadata(query, max_candidates=max_candidates)

@mcp.tool()
def execute_sql(query: str) -> str:
    """
    Executes a safe SELECT SQL query and returns the results.
    Do not use this for DELETE, UPDATE, INSERT, DROP, or TRUNCATE.
    """
    try:
        result = execute_safe_sql(query)
        if result["success"]:
            # Format results as string for llm
            data_str = str(result["data"])
            return f"Query executed successfully. Row count: {result['row_count']}\nResults: {data_str}"
        return f"Query execution failed unexpectedly: {result}"
    except Exception as e:
        return f"Error executing query: {str(e)}"

@mcp.tool()
def seed_data(table: str, rows: List[Dict[str, Any]]) -> str:
    """
    Inserts test or synthetic data into the specified database table.
    """
    try:
        result = seed_table(table, rows)
        if result["success"]:
            return f"Successfully inserted {result['inserted_rows']} rows into {result['table']}."
        return f"Failed to insert data: {result.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error seeding data: {str(e)}"

@mcp.tool()
def validate_data(data: List[Dict[str, Any]]) -> str:
    """
    Validates data structure before insertion. 
    Checks that the list is not empty and each row is a valid dictionary.
    """
    if not data:
        return "Validation Failed: Data list is empty."
    
    expected_keys = set(data[0].keys())
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            return f"Validation Failed: Row {i} is not a dictionary."
        if set(row.keys()) != expected_keys:
            return f"Validation Failed: Row {i} keys {set(row.keys())} do not match expected keys {expected_keys}."
            
    return "Validation Passed: Data structure is consistent."

@mcp.tool()
def update_documentation(content: str, filename: str = "schema_docs.md") -> str:
    """
    Updates system documentation files with the provided content.
    """
    try:
        os.makedirs("docs", exist_ok=True)
        file_path = os.path.join("docs", filename)
        
        mode = "a" if os.path.exists(file_path) else "w"
        
        with open(file_path, mode) as f:
            if mode == "a":
                f.write("\n\n")
            f.write(content)
            
        return f"Successfully updated documentation in {file_path}"
    except Exception as e:
        return f"Error updating documentation: {str(e)}"

if __name__ == "__main__":
    # When run directly, start the MCP server using standard I/O (stdio)
    mcp.run()
