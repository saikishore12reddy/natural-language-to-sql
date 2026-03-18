import asyncio
import json
from api.main import process_natural_language_query, QueryRequest

async def run_ambiguity_tests():
    print("--- Running Ambiguity & Fallback Tests ---")
    
    test_queries = [
        {
            "name": "1. High Confidence (Success)",
            "query": "Show all customers from Bangalore"
        },
        {
            "name": "2. Ambiguous (Clarification)",
            "query": "Show top customers"
        },
        {
            "name": "3. Fuzzy Schema Match (Error/Suggestions)",
            "query": "Show credits"
        },
        {
            "name": "4. Out of Scope (Error)",
            "query": "What is the capital of France?"
        },
        {
            "name": "5. Multi-intent (Clarification)",
            "query": "Show top customers and add test data"
        }
    ]
    
    for test in test_queries:
        print(f"\n>>> Name: {test['name']}")
        print(f">>> Query: {test['query']}")
        
        req = QueryRequest(query=test['query'])
        try:
            res = await process_natural_language_query(req)
            print(f"Response Type: {res.type}")
            print(f"Message: {res.message}")
            if res.rewritten_query:
                print(f"Rewritten Query: {res.rewritten_query}")
            if res.intent_analysis:
                print(f"Intent Analysis: {json.dumps(res.intent_analysis, indent=2)}")
            if res.options:
                print(f"Options: {res.options}")
            if res.suggestions:
                print(f"Suggestions: {res.suggestions}")
            if res.sql:
                print(f"SQL Executed: {res.sql}")
            if res.data:
                print(f"Data count: {len(res.data)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_ambiguity_tests())
