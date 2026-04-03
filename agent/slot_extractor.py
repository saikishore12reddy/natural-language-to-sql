"""
Slot Extraction Engine
======================
Extracts structured slots from a user's natural language query:
  - entity    : table to query (e.g. "loans")
  - attribute : column / field (e.g. "amount")
  - condition : filter expression (e.g. "< 2000")
  - operation : SQL operation type (e.g. "SELECT")

A query is marked *incomplete* when the entity or attribute slot is
missing AND the query implies a filtering or data-retrieval action.

Design
------
1. Rule-based shortcut  — catches obviously empty / single-token queries
   without touching the LLM.
2. LLM-based extraction — passes schema context so the model can resolve
   implicit attributes (e.g. "amount" → loans.amount).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SlotResult:
    entity: Optional[str]
    attribute: Optional[str]
    condition: Optional[str]
    operation: str
    status: str          # "complete" | "incomplete"
    missing: List[str]   # names of missing required slots

    def is_incomplete(self) -> bool:
        return self.status == "incomplete"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "attribute": self.attribute,
            "condition": self.condition,
            "operation": self.operation,
            "status": self.status,
            "missing": self.missing,
        }


# ---------------------------------------------------------------------------
# Rule-based shortcut helpers
# ---------------------------------------------------------------------------

# Patterns that strongly suggest a query is structurally incomplete
_INCOMPLETE_PATTERNS = [
    re.compile(r"^\s*$"),                          # empty / whitespace only
    re.compile(r"^\s*\w+\s*$"),                    # single word
    re.compile(r"^\d[\d\s]*$"),                    # pure numeric
    re.compile(                                    # comparison without subject
        r"^\s*(less\s+than|greater\s+than|more\s+than|>|<|>=|<=|!=)\s*$",
        re.IGNORECASE,
    ),
    re.compile(                                    # comparison with value but no noun
        r"^\s*(less\s+than|greater\s+than|more\s+than|>|<|>=|<=|!=)\s+[\d.]+\s*$",
        re.IGNORECASE,
    ),
    re.compile(                                    # vague open-ended phrases
        r"^\s*(show\s+me|give\s+me|get|fetch|find|top|bottom|all\s+data|data)\s*$",
        re.IGNORECASE,
    ),
]

def _rule_based_check(query: str) -> Optional[SlotResult]:
    """
    Quickly returns a SlotResult for obviously incomplete queries.
    Returns None if the rule-based check is inconclusive.
    """
    for pattern in _INCOMPLETE_PATTERNS:
        if pattern.match(query.strip()):
            return SlotResult(
                entity=None,
                attribute=None,
                condition=None,
                operation="SELECT",
                status="incomplete",
                missing=["entity", "attribute", "condition"],
            )
    return None


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are an advanced slot extraction engine for a Natural Language to SQL system.

Given a user query and a database schema, extract the following slots:
- "entity"    : The table the user is asking about (must be from the schema). Null if absent.
- "attribute" : The column/field being filtered or selected. Null if absent.
- "condition" : The filter expression (e.g. "< 2000", "= 'Bangalore'"). Null if absent or incomplete.
- "operation" : SQL operation implied ("SELECT", "COUNT", "SUM", "AVG", etc.). Default "SELECT".
- "confidence": float between 0.0 and 1.0 indicating confidence in the extraction.

Rules for Schema Matching:
1. ONLY use table names and column names that actually exist in the provided schema.
2. If the user mentions a synonym or alternative term (e.g., "client" for "customer"), map it to the closest matching schema entity.
3. For entity inference, consider:
   - Direct mentions: "show me customers" -> entity="customers"
   - Implicit queries: "all loans over 1000" -> entity="loans" (inferred from context)
   - Aggregations: "total amount" might need entity from context
4. For attribute inference:
   - "show me the amount" -> attribute="amount" (if unique across tables)
   - "how many" -> attribute is not needed (implies COUNT(*))
   - "the city" -> attribute="city"
5. A query is "complete" if:
   - Entity is identified (non-null) AND
   - Either: no filtering attribute needed (e.g., "list all customers") OR
   - Attribute is identified when a filter/comparison is present OR
   - Operation is an aggregation that doesn't require a specific attribute

Missing Slot Rules:
- Always include "entity" in missing if entity is null or low confidence (<0.6)
- Include "attribute" in missing if: (a) attribute is null AND (b) the query contains comparison words (>, <, =, greater, less, more, than, equal) or filter words (where, with, having)
- Include "condition" in missing if: condition is null AND query contains comparisons or "specific value" phrases

Return ONLY a JSON object with these exact keys:
{
  "entity": string or null,
  "attribute": string or null,
  "condition": string or null,
  "operation": string,
  "confidence": float,
  "status": "complete" or "incomplete",
  "missing": ["entity", "attribute", "condition"]  // subset of these
}

Examples:
Query: "show me customers" with schema having table "customers"
-> {"entity": "customers", "attribute": null, "condition": null, "operation": "SELECT", "confidence": 0.95, "status": "complete", "missing": []}

Query: "amount greater than 1000" with schema having tables "loans" (with amount) and "payments" (with amount)
-> {"entity": null, "attribute": "amount", "condition": "> 1000", "operation": "SELECT", "confidence": 0.3, "status": "incomplete", "missing": ["entity"]}

Query: "how many people in Bangalore" with schema having tables "customers" (name, city), "employees" (name, city)
-> {"entity": "customers", "attribute": null, "condition": "city = 'Bangalore'", "operation": "COUNT", "confidence": 0.8, "status": "complete", "missing": []}
"""


class SlotExtractor:
    """
    Extracts slots from a user query using a two-phase approach:
    1. Rule-based shortcut for obviously incomplete queries.
    2. LLM call with schema context for nuanced extraction.
    """

    def __init__(self, openai_client):
        """
        :param openai_client: An async OpenAI-compatible client (e.g. AsyncOpenAI pointed at Groq).
        """
        self._client = openai_client

    async def extract(
        self,
        query: str,
        schema_dict: Dict[str, List[str]],
        model: str = "llama-3.3-70b-versatile",
    ) -> SlotResult:
        """
        Extract slots from *query* using schema context.

        :param query:       Raw user input.
        :param schema_dict: {table_name: [col1, col2, ...]} from the database.
        :param model:       LLM model name.
        :returns:           A SlotResult instance.
        """
        # --- Phase 1: rule-based shortcut ---
        rule_result = _rule_based_check(query)
        if rule_result is not None:
            logger.warning(
                "Slot extraction (rule-based): incomplete | missing=%s | query=%r",
                rule_result.missing,
                query,
            )
            return rule_result

        # --- Phase 2: LLM extraction ---
        schema_text = self._format_schema(schema_dict)
        user_message = (
            f"Database schema:\n{schema_text}\n\n"
            f"User query: {query}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.error("LLM slot extraction failed: %s", exc)
            # Fail open — treat as complete so the rest of the pipeline handles it
            return SlotResult(
                entity=None,
                attribute=None,
                condition=None,
                operation="SELECT",
                status="incomplete",
                missing=["entity"],
            )

        result = SlotResult(
            entity=raw.get("entity") or None,
            attribute=raw.get("attribute") or None,
            condition=raw.get("condition") or None,
            operation=raw.get("operation") or "SELECT",
            status=raw.get("status", "incomplete"),
            missing=raw.get("missing") or [],
        )

        # Normalise status in case LLM returned inconsistent data
        if result.entity is None and "entity" not in result.missing:
            result.missing.append("entity")
        if result.missing:
            result.status = "incomplete"

        if result.is_incomplete():
            logger.warning(
                "Slot extraction (LLM): incomplete | missing=%s | query=%r",
                result.missing,
                query,
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_schema(schema_dict: Dict[str, List[str]]) -> str:
        lines = []
        for table, columns in schema_dict.items():
            lines.append(f"Table: {table}")
            lines.append(f"  Columns: {', '.join(columns)}")
        return "\n".join(lines) if lines else "No schema available."
