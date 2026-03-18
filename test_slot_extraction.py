"""
Test: Slot Extraction & Clarification Engine
============================================
Tests the new slot-extraction phase by calling the FastAPI endpoint handler
directly (no HTTP server needed), matching the pattern in test_ambiguity.py.

Run from the project root with the virtual environment active:
    source .venv/bin/activate
    python test_slot_extraction.py
"""

import asyncio
import json
from api.main import process_natural_language_query, QueryRequest

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

INCOMPLETE_QUERIES = [
    {
        "name": "1. Comparison fragment (no entity, no attribute)",
        "query": "show me less than",
        "expect_type": "clarification",
        "expect_missing_contains": ["entity", "attribute"],
    },
    {
        "name": "2. Value only (no entity, no attribute)",
        "query": "less than 2000",
        "expect_type": "clarification",
        "expect_missing_contains": ["entity"],
    },
    {
        "name": "3. Single ambiguous word",
        "query": "top",
        "expect_type": "clarification",
        "expect_missing_contains": ["entity"],
    },
    {
        "name": "4. Vague open-ended phrase",
        "query": "give me data",
        "expect_type": "clarification",
        "expect_missing_contains": ["entity"],
    },
    # Note: the FastAPI endpoint raises HTTP 400 for whitespace-only queries
    # BEFORE the slot extractor runs. This is correct guardrail behaviour.
    # The slot extractor itself handles single-word / near-empty queries (tests 1-4).
]

COMPLETE_QUERIES = [
    {
        "name": "6. Complete filter query — should reach SQL generation",
        "query": "show loans less than 2000",
        "expect_type": "success",
    },
    {
        "name": "7. Full customer query — should reach SQL generation",
        "query": "Show all customers from Bangalore",
        "expect_type": "success",
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PASS = "✅ PASS"
FAIL = "❌ FAIL"


async def run_tests():
    print("=" * 60)
    print("  Slot Extraction & Clarification Engine — Test Suite")
    print("=" * 60)

    all_passed = True

    # --- Incomplete query tests ---
    print("\n📋 INCOMPLETE QUERY TESTS (expect: clarification)\n")
    for tc in INCOMPLETE_QUERIES:
        print(f"  >>> {tc['name']}")
        print(f"      Query: {tc['query']!r}")
        req = QueryRequest(query=tc["query"])
        try:
            res = await process_natural_language_query(req)
        except Exception as e:
            print(f"      {FAIL} — Exception: {e}\n")
            all_passed = False
            continue

        type_ok = res.type == tc["expect_type"]
        missing_ok = all(
            s in (res.missing_slots or [])
            for s in tc.get("expect_missing_contains", [])
        )
        passed = type_ok and missing_ok

        status = PASS if passed else FAIL
        print(f"      {status}")
        print(f"      Response type : {res.type}")
        print(f"      Missing slots : {res.missing_slots}")
        print(f"      Message       : {res.message}")
        if res.options:
            for opt in res.options:
                print(f"        • {opt}")
        if res.slot_analysis:
            print(f"      Slot analysis : {json.dumps(res.slot_analysis, indent=8)}")
        print()
        if not passed:
            all_passed = False

    # --- Complete query tests ---
    print("\n📋 COMPLETE QUERY TESTS (expect: success)\n")
    for tc in COMPLETE_QUERIES:
        print(f"  >>> {tc['name']}")
        print(f"      Query: {tc['query']!r}")
        req = QueryRequest(query=tc["query"])
        try:
            res = await process_natural_language_query(req)
        except Exception as e:
            print(f"      {FAIL} — Exception: {e}\n")
            all_passed = False
            continue

        passed = res.type == tc["expect_type"]
        status = PASS if passed else FAIL
        print(f"      {status}")
        print(f"      Response type : {res.type}")
        if res.sql:
            print(f"      SQL executed  : {res.sql}")
        if res.data is not None:
            print(f"      Row count     : {res.row_count}")
        if res.message:
            print(f"      Message       : {res.message[:120]}")
        print()
        if not passed:
            all_passed = False

    # --- Summary ---
    print("=" * 60)
    if all_passed:
        print("  🎉  All tests passed!")
    else:
        print("  ⚠️   Some tests FAILED — review output above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())
