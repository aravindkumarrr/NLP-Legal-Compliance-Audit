"""
Stage 4: Deontic Logic Knowledge Graph construction (Neo4j).

Builds the graph described in the design figure: every clause becomes a
node labelled by its deontic category (Obligation / Permission /
Prohibition / Penalty), linked to the Entity it binds via APPLIES_TO,
and -- the relationship the whole restructuring was for -- a Prohibition
or Obligation node linked to the Penalty node that applies if it's
violated, via PENALIZED_BY.

Penalty linkage is the one part that can't come from the LLM classifier
alone (it wasn't asked to find cross-references), so it's done with a
lightweight regex pass over clause text looking for explicit
cross-references ("Section 3.2", "clause (a)", "this Clause"), matched
against clause labels already captured by segmentation.py. This is a
heuristic, not a guarantee -- documents that describe penalties without
an explicit clause reference won't get linked, which is a known
limitation worth calling out in the report/demo rather than papering
over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from neo4j import Driver

from app.pipeline.deontic import DeonticCategory, DeonticLabel
from app.pipeline.segmentation import Clause


@dataclass
class GraphBuildResult:
    nodes_created: int
    edges_created: int


# Matches cross-references like "Section 3.2", "Clause 5", "clause (a)", "this Clause 4.1"
_REFERENCE_PATTERN = re.compile(
    r"""(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*|\([a-zA-Z0-9]+\))""",
    re.IGNORECASE,
)


def _find_referenced_labels(text: str) -> set[str]:
    labels = set()
    for match in _REFERENCE_PATTERN.finditer(text):
        label = match.group(1).strip("()")
        labels.add(label)
    return labels


def build_graph_for_document(
    driver: Driver,
    document_id: str,
    clauses: list[Clause],
    labels: list[DeonticLabel],
) -> GraphBuildResult:
    """
    Wipes and rebuilds this document's subgraph (idempotent re-runs), then
    creates one node per clause, one node per distinct entity, APPLIES_TO
    edges between them, and PENALIZED_BY edges between
    Obligation/Prohibition nodes and any Penalty node whose clause text
    cross-references them.
    """
    clause_by_id = {c.id: c for c in clauses}
    label_by_clause_id = {l.clause_id: l for l in labels}
    # Map detected clause *label* (e.g. "3.2") -> clause id, so we can
    # resolve "Section 3.2" cross-references back to a node.
    clause_id_by_number_label = {c.label: c.id for c in clauses if c.label}

    nodes_created = 0
    edges_created = 0

    with driver.session() as session:
        session.execute_write(_clear_document_subgraph, document_id)

        for clause in clauses:
            deontic = label_by_clause_id.get(clause.id)
            category = deontic.category if deontic else DeonticCategory.UNKNOWN
            entity = (deontic.entity if deontic and deontic.entity else "unspecified").strip()

            session.execute_write(
                _create_clause_node, document_id, clause, category, entity
            )
            nodes_created += 1

            session.execute_write(_create_entity_edge, document_id, clause.id, entity)
            edges_created += 1

        # Second pass: penalty linkage needs all nodes to already exist.
        for clause in clauses:
            deontic = label_by_clause_id.get(clause.id)
            if deontic is None or deontic.category != DeonticCategory.PENALTY:
                continue

            referenced_labels = _find_referenced_labels(clause.text)
            for ref_label in referenced_labels:
                target_clause_id = clause_id_by_number_label.get(ref_label)
                if target_clause_id is None or target_clause_id == clause.id:
                    continue
                target_label = label_by_clause_id.get(target_clause_id)
                if target_label is None or target_label.category not in (
                    DeonticCategory.OBLIGATION,
                    DeonticCategory.PROHIBITION,
                ):
                    continue

                session.execute_write(
                    _create_penalty_edge, document_id, target_clause_id, clause.id
                )
                edges_created += 1

    return GraphBuildResult(nodes_created=nodes_created, edges_created=edges_created)


def _clear_document_subgraph(tx, document_id: str) -> None:
    tx.run(
        "MATCH (n {document_id: $document_id}) DETACH DELETE n",
        document_id=document_id,
    )


def _create_clause_node(tx, document_id: str, clause: Clause, category: DeonticCategory, entity: str) -> None:
    # Category becomes a Neo4j label so Cypher queries can filter by it
    # directly (MATCH (n:Obligation) ...) rather than a property lookup.
    tx.run(
        f"""
        MERGE (c:Clause:{category.value} {{document_id: $document_id, clause_id: $clause_id}})
        SET c.label = $label, c.text = $text, c.order = $order, c.entity = $entity
        """,
        document_id=document_id,
        clause_id=clause.id,
        label=clause.label,
        text=clause.text,
        order=clause.order,
        entity=entity,
    )


def _create_entity_edge(tx, document_id: str, clause_id: str, entity: str) -> None:
    tx.run(
        """
        MERGE (e:Entity {document_id: $document_id, name: $entity})
        WITH e
        MATCH (c:Clause {document_id: $document_id, clause_id: $clause_id})
        MERGE (c)-[:APPLIES_TO]->(e)
        """,
        document_id=document_id,
        clause_id=clause_id,
        entity=entity,
    )


def _create_penalty_edge(tx, document_id: str, source_clause_id: str, penalty_clause_id: str) -> None:
    tx.run(
        """
        MATCH (source:Clause {document_id: $document_id, clause_id: $source_clause_id})
        MATCH (penalty:Clause:Penalty {document_id: $document_id, clause_id: $penalty_clause_id})
        MERGE (source)-[:PENALIZED_BY]->(penalty)
        """,
        document_id=document_id,
        source_clause_id=source_clause_id,
        penalty_clause_id=penalty_clause_id,
    )


def get_related_clauses(driver: Driver, document_id: str, clause_id: str, hops: int = 1) -> list[dict]:
    """
    Used by graphrag.py: returns clauses within `hops` graph steps of the
    given clause via APPLIES_TO or PENALIZED_BY (in either direction) --
    e.g. an Obligation's linked Penalty, or other clauses binding the
    same Entity.
    """
    with driver.session() as session:
        result = session.execute_read(_query_related_clauses, document_id, clause_id, hops)
    return result


def _query_related_clauses(tx, document_id: str, clause_id: str, hops: int) -> list[dict]:
    records = tx.run(
        f"""
        MATCH (start:Clause {{document_id: $document_id, clause_id: $clause_id}})
        MATCH (start)-[:APPLIES_TO|PENALIZED_BY*1..{hops}]-(related:Clause)
        WHERE related.clause_id <> $clause_id
        RETURN DISTINCT related.clause_id AS clause_id, related.text AS text,
               related.label AS label, labels(related) AS labels
        """,
        document_id=document_id,
        clause_id=clause_id,
    )
    return [dict(r) for r in records]
