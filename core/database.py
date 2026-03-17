import os
import re
from typing import List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})

def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validates a SQL query to ensure it only performs safe read operations.
    Returns (is_valid, error_message).
    """
    # Convert to uppercase for checking
    q_upper = query.upper()
    
    # Block unsafe commands
    unsafe_keywords = [
        "DELETE", "UPDATE", "INSERT", "DROP", "TRUNCATE", 
        "ALTER", "CREATE", "GRANT", "REVOKE", "EXECUTE"
    ]
    
    # Check if the query starts with an unsafe keyword or contains it in a dangerous way
    # Simple regex to check for whole word matches of unsafe keywords
    for keyword in unsafe_keywords:
        if re.search(r'\b' + keyword + r'\b', q_upper):
            # Allow INSERT only if specifically called by a different tool, but for execute_sql it's blocked.
            # We strictly enforce SELECT only for this validator
            return False, f"Unsafe SQL keyword detected: {keyword}. Only SELECT queries are permitted."
            
    # Require SELECT
    if not re.search(r'\bSELECT\b', q_upper):
        return False, "Only SELECT queries are allowed."
        
    return True, ""


def execute_safe_sql(query: str) -> Dict[str, Any]:
    """
    Executes a SQL query after validating it's safe.
    Returns the results as a list of dicts.
    """
    is_valid, error_message = validate_query(query)
    if not is_valid:
        raise ValueError(f"Query validation failed: {error_message}")
    
    # Enforce LIMIT if not present (simple text check, robust enough for basic protection)
    if not re.search(r'\bLIMIT\b', query.upper()):
        query = f"{query} LIMIT 100"
        
    with engine.connect() as connection:
        result = connection.execute(text(query))
        
        # Determine if it's a DDL/DML that returns no rows or a SELECT that returns rows
        if result.returns_rows:
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in result.fetchall()]
            return {
                "success": True,
                "data": data,
                "row_count": len(data),
                "executed_query": query
            }
        else:
            return {
                "success": True,
                "data": [],
                "row_count": 0,
                "executed_query": query
            }

def seed_table(table_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Securely inserts test data into a table.
    """
    if not rows:
        return {"success": False, "message": "No data provided for seeding."}
        
    columns = list(rows[0].keys())
    columns_str = ", ".join(columns)
    
    # Using parameterized queries to prevent SQL injection during seed
    params_str = ", ".join([f":{col}" for col in columns])
    
    query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({params_str})"
    
    with engine.begin() as connection:
        result = connection.execute(text(query), rows)
        
    return {
        "success": True,
        "inserted_rows": len(rows),
        "table": table_name
    }
