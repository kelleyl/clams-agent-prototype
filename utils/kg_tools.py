"""Knowledge graph query tools for the LangGraph agent.

Wraps kg.graph.KnowledgeGraph methods into agent-friendly functions
that return structured results matching the tool_history pattern.
"""

import json
import logging
from pathlib import Path

from kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_DB = DATA_DIR / "knowledge_graph.db"

_kg_instance: KnowledgeGraph | None = None


def get_kg() -> KnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph(DEFAULT_DB)
    return _kg_instance


def kg_search(query: str, entity_type: str | None = None, strict: bool = True) -> dict:
    """Search entities by name. Returns matches with types and relation counts.

    Uses strict matching by default — exact/case-insensitive match, then
    word-overlap (e.g. "Hillary Rodham" matches "Hillary Rodham Clinton").
    No substring matching to avoid false positives.
    """
    kg = get_kg()
    results = kg.search_entity(query, strict=strict)

    # Fall back to non-strict if strict finds nothing
    if not results and strict:
        results = kg.search_entity(query, strict=False)

    if entity_type:
        # Normalize type aliases
        type_aliases = {
            "university": "organization", "company": "organization",
            "country": "location", "city": "location", "place": "location",
            "people": "person", "individual": "person",
        }
        normalized_type = type_aliases.get(entity_type.lower(), entity_type.lower())
        results = [r for r in results if r.get("entity_type") == normalized_type]

    # Enrich with relation counts
    enriched = []
    for r in results[:20]:
        rels = kg.get_relations(r["entity_id"])
        # Group relations by predicate
        rel_summary = {}
        for rel in rels:
            pred = rel["predicate"]
            rel_summary[pred] = rel_summary.get(pred, 0) + 1

        enriched.append({
            "entity_id": r["entity_id"],
            "name": r["canonical_name"],
            "type": r["entity_type"],
            "qid": r.get("wikidata_qid") or "",
            "relation_counts": rel_summary,
        })

    summary = f"Found {len(enriched)} entities matching '{query}'"
    if enriched:
        top_names = [e["name"] for e in enriched[:5]]
        summary += f": {', '.join(top_names)}"
        if len(enriched) > 5:
            summary += f" (+{len(enriched) - 5} more)"

    return {
        "success": True,
        "result_summary": summary,
        "detail": _format_entities(enriched),
        "entities": enriched,
    }


def kg_neighbors(
    entity_id: str,
    hops: int = 1,
    predicate_filter: str | list | None = None,
) -> dict:
    """Explore entity neighborhood — relations and connected entities."""
    kg = get_kg()

    entity = kg.get_entity(entity_id)
    if not entity:
        # Try searching by name
        results = kg.search_entity(entity_id)
        if results:
            entity = results[0]
            entity_id = entity["entity_id"]
        else:
            return {
                "success": False,
                "result_summary": f"Entity '{entity_id}' not found",
                "detail": "",
                "entities": [],
            }

    # Normalize predicate_filter: accept list or string
    if isinstance(predicate_filter, list):
        predicate_filter = predicate_filter[0] if predicate_filter else None

    rels = kg.get_relations(entity_id, predicate=predicate_filter)

    # Group by predicate for cleaner output
    by_predicate: dict[str, list] = {}
    for rel in rels:
        pred = rel["predicate"]
        if pred not in by_predicate:
            by_predicate[pred] = []

        # Resolve the other entity
        other_id = rel["object_id"] if rel["subject_id"] == entity_id else rel["subject_id"]
        other = kg.get_entity(other_id)
        other_name = other["canonical_name"] if other else other_id
        direction = "→" if rel["subject_id"] == entity_id else "←"

        by_predicate[pred].append({
            "entity_id": other_id,
            "name": other_name,
            "type": other.get("entity_type", "") if other else "",
            "direction": direction,
            "evidence": rel.get("evidence") or "",
            "video_id": rel.get("video_id") or "",
        })

    # Build output
    detail_lines = [f"Entity: {entity['canonical_name']} ({entity.get('entity_type', '')})"]
    if entity.get("wikidata_qid"):
        detail_lines.append(f"QID: {entity['wikidata_qid']}")

    # Add external facts if available
    if entity.get("external_facts_json"):
        try:
            facts = json.loads(entity["external_facts_json"])
            if isinstance(facts, dict):
                for k, v in list(facts.items())[:8]:
                    val = ", ".join(v) if isinstance(v, list) else str(v)
                    detail_lines.append(f"  {k}: {val}")
        except (json.JSONDecodeError, TypeError):
            pass

    detail_lines.append("")

    all_connected = []
    for pred, items in sorted(by_predicate.items()):
        detail_lines.append(f"  {pred} ({len(items)}):")
        for item in items[:10]:
            detail_lines.append(
                f"    {item['direction']} {item['name']} ({item['type']})"
                + (f"  [{item['evidence'][:60]}]" if item["evidence"] else "")
            )
            all_connected.append(item)
        if len(items) > 10:
            detail_lines.append(f"    ... and {len(items) - 10} more")

    summary = (
        f"{entity['canonical_name']}: {len(rels)} relations "
        f"across {len(by_predicate)} types"
    )

    return {
        "success": True,
        "result_summary": summary,
        "detail": "\n".join(detail_lines),
        "entities": all_connected,
        "center_entity": {
            "entity_id": entity_id,
            "name": entity["canonical_name"],
            "type": entity.get("entity_type", ""),
        },
    }


def kg_path(entity1: str, entity2: str, max_hops: int = 4) -> dict:
    """Find shortest path between two entities."""
    kg = get_kg()

    # Resolve names to IDs if needed
    e1_id = _resolve_entity_id(kg, entity1)
    e2_id = _resolve_entity_id(kg, entity2)

    if not e1_id:
        return {"success": False, "result_summary": f"Entity '{entity1}' not found", "detail": ""}
    if not e2_id:
        return {"success": False, "result_summary": f"Entity '{entity2}' not found", "detail": ""}

    path = kg.find_path(e1_id, e2_id, max_hops)
    if not path:
        return {
            "success": True,
            "result_summary": f"No path found between '{entity1}' and '{entity2}' within {max_hops} hops",
            "detail": "",
        }

    # Format path
    detail_lines = ["Path:"]
    hop_count = 0
    for item in path:
        if item["type"] == "entity":
            detail_lines.append(f"  [{item['canonical_name']}] ({item.get('entity_type', '')})")
        elif item["type"] == "relation":
            hop_count += 1
            detail_lines.append(f"    --{item['predicate']}--> (source: {item.get('source', '')})")

    e1_name = path[0].get("canonical_name", entity1) if path else entity1
    e2_name = path[-1].get("canonical_name", entity2) if path else entity2

    return {
        "success": True,
        "result_summary": f"Path found: {e1_name} → {e2_name} ({hop_count} hops)",
        "detail": "\n".join(detail_lines),
        "path": path,
    }


def kg_query(
    predicate: str,
    value: str | None = None,
    subject_type: str | None = None,
) -> dict:
    """Find entities by relation type and value.

    Examples:
      kg_query("occupation", "lawyer") → all people whose occupation is lawyer
      kg_query("political_party", "Republican Party") → Republicans
      kg_query("educated_at", subject_type="person") → all people with education info
    """
    kg = get_kg()
    conn = kg._connect()

    sql = """
        SELECT DISTINCT e.entity_id, e.canonical_name, e.entity_type, e.wikidata_qid,
               r.predicate, obj.canonical_name as object_name
        FROM relations r
        JOIN entities e ON e.entity_id = r.subject_id
        JOIN entities obj ON obj.entity_id = r.object_id
        WHERE r.predicate = ?
    """
    params: list = [predicate]

    if value:
        sql += " AND UPPER(obj.canonical_name) LIKE UPPER(?)"
        params.append(f"%{value}%")

    if subject_type:
        sql += " AND e.entity_type = ?"
        params.append(subject_type.lower())

    sql += " ORDER BY e.canonical_name LIMIT 50"

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "entity_id": r["entity_id"],
            "name": r["canonical_name"],
            "type": r["entity_type"],
            "qid": r["wikidata_qid"] or "",
            "predicate": r["predicate"],
            "value": r["object_name"],
        })

    if value:
        summary = f"Found {len(results)} entities with {predicate} = '{value}'"
    else:
        summary = f"Found {len(results)} entities with {predicate} relations"

    detail_lines = []
    for r in results[:30]:
        detail_lines.append(f"  {r['name']} ({r['type']}) → {r['predicate']}: {r['value']}")

    return {
        "success": True,
        "result_summary": summary,
        "detail": "\n".join(detail_lines),
        "entities": results,
    }


def _resolve_entity_id(kg: KnowledgeGraph, name_or_id: str) -> str | None:
    """Resolve a name or ID to an entity_id."""
    entity = kg.get_entity(name_or_id)
    if entity:
        return name_or_id

    results = kg.search_entity(name_or_id)
    if results:
        return results[0]["entity_id"]
    return None


def _format_entities(entities: list[dict]) -> str:
    """Format entity list for display."""
    lines = []
    for e in entities:
        line = f"  {e['name']} ({e['type']})"
        if e.get("qid"):
            line += f" [QID: {e['qid']}]"
        if e.get("relation_counts"):
            counts = ", ".join(f"{k}:{v}" for k, v in sorted(e["relation_counts"].items()))
            line += f"\n    relations: {counts}"
        lines.append(line)
    return "\n".join(lines)
