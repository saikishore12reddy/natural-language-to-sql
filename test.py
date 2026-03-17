import asyncio
import httpx
from core.database import execute_safe_sql, engine
from sqlalchemy import text
import pprint

async def run_tests():
    print("--- Setting up test tables ---")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                city VARCHAR(50)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS loans (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                amount DECIMAL,
                status VARCHAR(20),
                FOREIGN KEY(customer_id) REFERENCES customers(id)
            )
        """))
    print("Tables created.")

    print("\n--- Testing API Endpoint (FastAPI) ---")
    # We will just test the agent directly instead of the API to avoid running a background server
    # for this test, but let's test the agent functionality through the FastAPI app's logic
    from api.main import process_natural_language_query, QueryRequest
    
    print("\n1. Testing Schema Inspector (Agent)")
    req = QueryRequest(query="What tables are in the database?")
    res = await process_natural_language_query(req)
    print("Agent Response:", res.message)
    
    print("\n2. Testing Data Seeding")
    req = QueryRequest(query="Add 3 test customers in Bangalore and 2 in Mumbai. Then add 5 loans for them with different amounts.")
    res = await process_natural_language_query(req)
    print("Agent Response:", res.message)
    
    print("\n3. Testing Natural Language Query Interface")
    req = QueryRequest(query="Show top 3 customers with highest loan amount")
    res = await process_natural_language_query(req)
    print("Agent Response (SQL executed):", res.sql)
    print("Agent Data:", res.data)
    
    print("\n4. Testing Documentation Update")
    req = QueryRequest(query="Please update the documentation with the current database schema")
    res = await process_natural_language_query(req)
    print("Agent Response:", res.message)

if __name__ == "__main__":
    asyncio.run(run_tests())
