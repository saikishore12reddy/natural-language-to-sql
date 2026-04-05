#!/usr/bin/env python3
"""
Interactive query clarification client.

This script sends queries to the NL2SQL API and handles clarification interactively.

Usage:
    python query_clarifier.py "your query here"
"""

import sys
import json
import asyncio
import aiohttp


async def process_query_with_clarification(api_url: str, original_query: str) -> dict:
    """Sends a query to the API and handles clarification interactively."""
    current_query = original_query.strip()

    async with aiohttp.ClientSession() as session:
        while True:
            # Send current query to API
            async with session.post(
                f"{api_url.rstrip('/')}/query",
                json={"query": current_query}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return {"type": "error", "message": f"API error: {error}"}

                result = await resp.json()

            # If no clarification needed, return the result
            if result.get("type") != "clarification":
                return result

            # Display clarification prompt and options
            print("\n" + "="*70)
            print("CLARIFICATION NEEDED")
            print("="*70)
            print(result.get("message", "Your query needs more details."))
            print("\nPlease choose one of the following options:")
            options = result.get("options", [])
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
            print("  q. Quit")

            # Get user selection with validation
            while True:
                choice = input("\nEnter choice number (or 'q' to quit): ").strip().lower()
                if choice == 'q':
                    return {"type": "cancelled", "message": "Clarification cancelled by user."}

                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(options):
                        selected_option = options[choice_idx]
                        break
                    else:
                        print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number or 'q' to quit.")

            print(f"\nYou selected: {selected_option}")
            print("Please provide any additional details (or press Enter to continue):")
            additional = input("> ").strip()

            # Refine the query using the selection and additional info
            refined_query = selected_option
            if additional:
                refined_query = f"{selected_option} - {additional}"

            print(f"\nRefined query: {refined_query}")
            print("\nSubmitting refined query...")

            # Continue loop: the refined query will be sent back to the API
            current_query = refined_query


async def main():
    if len(sys.argv) < 2:
        print("Usage: python query_clarifier.py \"your query here\"")
        print("Example: python query_clarifier.py \"Show\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    api_url = "http://localhost:8000"  # Default API URL

    print(f"Original query: {query}")
    print("Processing...")

    try:
        result = await process_query_with_clarification(api_url, query)

        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)

        if result.get("type") == "success":
            print(f"Status: SUCCESS")
            print(f"SQL: {result.get('sql', 'N/A')}")
            print(f"\nResults ({result.get('row_count', 0)} rows):")
            data = result.get('data', [])
            if data:
                for row in data[:10]:  # limit to first 10 rows
                    print(f"  {row}")
                if len(data) > 10:
                    print(f"  ... and {len(data) - 10} more rows")
            else:
                print("  No data returned")
        elif result.get("type") == "clarification":
            # Should not reach here if loop works correctly
            print("Unexpected clarification without options")
        elif result.get("type") == "cancelled":
            print("Operation cancelled.")
        else:
            print(f"Status: ERROR")
            print(f"Message: {result.get('message', 'Unknown error')}")

    except aiohttp.ClientError as e:
        print(f"Could not connect to API: {e}")
        print("Make sure the FastAPI server is running on http://localhost:8000")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())