"""
Clarification Engine
====================
Generates schema-aware, user-friendly clarification responses when a query
is detected as incomplete by the SlotExtractor.

The engine:
1. Inspects which slots are missing (entity, attribute, condition).
2. Calls the LLM with schema context to produce 2–3 concrete, actionable
   suggestions the user can pick from.
3. Returns a structured dict matching the API's clarification response shape.

All incomplete queries are logged for observability / metrics.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.slot_extractor import SlotResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLARIFICATION_SYSTEM_PROMPT = """\
You are a helpful assistant for a Natural Language to SQL system.

A user submitted an incomplete query. Your job is to:
1. Write a short, friendly clarification message explaining what is missing.
2. Generate 2-3 specific, actionable suggestions the user can select next.
   Each suggestion must be a complete, natural language query that the system
   can process (it should mention the table AND relevant column/filter).
   Base the suggestions ONLY on the tables and columns listed in the schema.

Return ONLY a JSON object with:
  "message"  : string — friendly clarification message
  "options"  : list of strings — 2-3 complete query suggestions
"""

_SLOT_LABELS: Dict[str, str] = {
    "entity": "the data you want to look at (e.g. loans, customers)",
    "attribute": "the column or field to filter by (e.g. amount, city)",
    "condition": "a complete filter value (e.g. less than 2000, equal to 'Mumbai')",
}


def _missing_description(missing: List[str]) -> str:
    """Human-readable description of which slots are absent."""
    if not missing:
        return "some details"
    labels = [_SLOT_LABELS.get(m, m) for m in missing]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


# ---------------------------------------------------------------------------
# ClarificationEngine
# ---------------------------------------------------------------------------

class ClarificationEngine:
    """
    Produces a clarification response dict from an incomplete SlotResult.
    """

    def __init__(self, openai_client):
        """
        :param openai_client: Async OpenAI-compatible client (e.g. Groq).
        """
        self._client = openai_client

    async def generate(
        self,
        query: str,
        slot_result: SlotResult,
        schema_dict: Dict[str, List[str]],
        model: str = "llama-3.3-70b-versatile",
    ) -> Dict[str, Any]:
        """
        Generate a clarification response for an incomplete query.

        :param query:       The original user query.
        :param slot_result: The SlotResult returned by SlotExtractor.
        :param schema_dict: {table: [col, ...]} for the connected database.
        :param model:       LLM model name.
        :returns:           Dict with type, message, missing, options, slot_analysis.
        """
        # Log for observability / metrics
        logger.warning(
            "Incomplete query | missing_slots=%s | query=%r",
            slot_result.missing,
            query,
        )

        # Build schema text for the prompt
        schema_text = self._format_schema(schema_dict)
        missing_desc = _missing_description(slot_result.missing)

        user_message = (
            f"Database schema:\n{schema_text}\n\n"
            f"User query: \"{query}\"\n\n"
            f"Missing slots: {', '.join(slot_result.missing) if slot_result.missing else 'unknown'}\n"
            f"Partial slot data: entity={slot_result.entity!r}, "
            f"attribute={slot_result.attribute!r}, "
            f"condition={slot_result.condition!r}\n\n"
            f"Generate a helpful clarification asking the user to specify: {missing_desc}."
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CLARIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            llm_data = json.loads(response.choices[0].message.content)
            message = llm_data.get("message", "Your query is incomplete. Could you please provide more details?")
            options = llm_data.get("options", self._fallback_options(schema_dict, slot_result))
        except Exception as exc:
            logger.error("ClarificationEngine LLM call failed: %s", exc)
            message = self._fallback_message(slot_result)
            options = self._fallback_options(schema_dict, slot_result)

        return {
            "type": "clarification",
            "message": message,
            "missing": slot_result.missing,
            "options": options[:3],   # cap at 3
            "slot_analysis": slot_result.to_dict(),
        }

    # ------------------------------------------------------------------
    # Fallbacks (used when LLM call fails)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_message(slot_result: SlotResult) -> str:
        if not slot_result.missing:
            return "Your query is incomplete. Could you rephrase it with more details?"
        missing_desc = _missing_description(slot_result.missing)
        return f"Your query is incomplete. Please specify {missing_desc}."

    @staticmethod
    def _fallback_options(
        schema_dict: Dict[str, List[str]],
        slot_result: SlotResult,
    ) -> List[str]:
        """
        Generate basic suggestions from the schema without using the LLM.
        """
        options = []
        for table, columns in list(schema_dict.items())[:2]:
            # Pick the first non-id numeric-sounding column, else first column
            col = next(
                (c for c in columns if c not in ("id",) and "id" not in c.lower()),
                columns[0] if columns else "value",
            )
            cond = slot_result.condition or "a specific value"
            options.append(f"Show {table} with {col} {cond}")
        if not options:
            options.append("Please provide more details about what you want to retrieve.")
        return options

    @staticmethod
    def _format_schema(schema_dict: Dict[str, List[str]]) -> str:
        lines = []
        for table, columns in schema_dict.items():
            lines.append(f"Table: {table}")
            lines.append(f"  Columns: {', '.join(columns)}")
        return "\n".join(lines) if lines else "No schema available."
