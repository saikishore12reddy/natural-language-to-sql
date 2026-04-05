import re
from sqlalchemy import inspect, text
from core.database import engine
import difflib
from typing import Dict, List, Tuple

def get_all_tables_info() -> Dict[str, Dict]:
    """
    Returns complete database schema as a nested dict:
    {
        "table_name": {
            "columns": [{"name": str, "type": str}, ...],
            "pk": [str, ...],
            "fks": [{"constrained_cols": [str, ...], "referred_table": str, "referred_cols": [str, ...]}, ...]
        }
    }
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    all_info: Dict[str, Dict] = {}

    for table_name in tables:
        columns = inspector.get_columns(table_name)
        pk_constraint = inspector.get_pk_constraint(table_name)
        fks = inspector.get_foreign_keys(table_name)

        all_info[table_name] = {
            "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
            "pk": pk_constraint.get("constrained_columns", []) if pk_constraint else [],
            "fks": [
                {
                    "constrained_cols": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_cols": fk["referred_columns"],
                }
                for fk in fks
            ],
        }

    return all_info


def _build_schema_string(filtered_info: Dict[str, Dict]) -> str:
    """Builds the human-readable schema string from filtered table info."""
    schema_lines = []

    for table_name in sorted(filtered_info.keys()):
        info = filtered_info[table_name]
        schema_lines.append(f"Table: {table_name}")

        col_details = [f"{c['name']} ({c['type']})" for c in info["columns"]]
        schema_lines.append(f"  Columns: {', '.join(col_details)}")

        if info["pk"]:
            schema_lines.append(f"  Primary Keys: {', '.join(info['pk'])}")

        for fk in info["fks"]:
            constrained = ", ".join(fk["constrained_cols"])
            referred = f"{fk['referred_table']}.{', '.join(fk['referred_cols'])}"
            schema_lines.append(f"  Foreign Key: {constrained} -> {referred}")

        schema_lines.append("")  # Empty line for spacing

    if not schema_lines:
        return "Database is currently empty with no tables."

    return "\n".join(schema_lines)


def get_schema_metadata() -> str:
    """
    Uses SQLAlchemy inspector to dynamically fetch database schema.
    Returns a formatted string representing tables and columns.
    """
    all_info = get_all_tables_info()
    return _build_schema_string(all_info)


def get_filtered_schema_metadata(query: str, max_candidates: int = 8, max_fk_hops: int = 1) -> str:
    """
    Returns schema metadata for only the tables relevant to the query.
    Includes FK-related tables to preserve join context.

    Args:
        query: User's natural language query
        max_candidates: Max number of candidate tables to initially identify
        max_fk_hops: How many FK relationship hops to traverse (default 1)

    Returns:
        Formatted schema string for filtered tables.
    """
    all_info = get_all_tables_info()
    table_names = list(all_info.keys())

    if not table_names:
        return "Database is currently empty with no tables."

    # Phase 1: Candidate identification via fuzzy matching
    candidates = _identify_candidates(query, table_names, max_candidates)

    # If we have fewer than 2 candidates, fall back to full schema
    if len(candidates) < 2:
        return _build_schema_string(all_info)

    # Phase 2: Expand via FK relationships
    related_tables = _expand_with_relationships(candidates, all_info, max_fk_hops)

    # Phase 3: Build schema string for final set
    filtered_info = {table: all_info[table] for table in sorted(related_tables) if table in all_info}

    return _build_schema_string(filtered_info)


def _identify_candidates(query: str, table_names: List[str], max_candidates: int) -> List[str]:
    """
    Identifies candidate tables by fuzzy matching table names in the query.
    Returns top N matches with similarity cutoff.
    """
    # Normalize query: lowercase, remove punctuation
    normalized = re.sub(r"[^\w\s]", "", query.lower())
    words = set(normalized.split())

    # Score tables by word similarity
    table_scores: List[Tuple[str, float]] = []
    for table in table_names:
        table_lower = table.lower()
        # Direct keyword match = high score
        if table_lower in words:
            table_scores.append((table, 1.0))
            continue

        # Fuzzy match against the full table name
        match_ratio = difflib.SequenceMatcher(None, normalized, table_lower).ratio()
        if match_ratio > 0.3:
            table_scores.append((table, match_ratio))

    # Sort by score descending and take top N
    table_scores.sort(key=lambda x: x[1], reverse=True)
    candidates = [t for t, score in table_scores[:max_candidates]]

    return candidates


def _expand_with_relationships(candidates: List[str], all_info: Dict[str, Dict], max_hops: int) -> set:
    """
    Expands the candidate set by including tables connected via foreign keys.
    Uses breadth-first expansion up to max_hops.
    """
    expanded = set(candidates)
    frontier = list(candidates)

    for _ in range(max_hops):
        next_frontier = []
        for table in frontier:
            info = all_info[table]
            # Add referred tables (where this table references them)
            for fk in info["fks"]:
                referred = fk["referred_table"]
                if referred not in expanded:
                    expanded.add(referred)
                    next_frontier.append(referred)
            # Also add referring tables (where other tables reference this one)
            for other, other_info in all_info.items():
                for fk in other_info["fks"]:
                    if fk["referred_table"] == table and other not in expanded:
                        expanded.add(other)
                        next_frontier.append(other)
        frontier = next_frontier
        if not frontier:
            break

    return expanded
