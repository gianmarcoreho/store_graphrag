"""Shared GraphRAG helpers: model factory, ontology, knowledge graph, utils."""

from __future__ import annotations

import json
import os
import socket
import time
from itertools import product
from typing import Optional

import pandas as pd

from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig, Ontology
from graphrag_sdk.helpers import extract_cypher, validate_cypher
from graphrag_sdk.models import GenerativeModelConfig
from graphrag_sdk.models.litellm import LiteModel

from . import config


def create_model() -> LiteModel:
    """Create the Qwen (vLLM) model used for both Cypher generation and Q&A.

    Qwen3.6 is a *reasoning* model: with thinking enabled it spends the whole
    token budget in the ``reasoning`` field and never emits ``content`` (the only
    field the SDK reads), so queries appear to hang. Disabling thinking makes
    responses fast (~1 s) and puts the final answer directly in ``content``.
    """
    os.environ["OPENAI_API_KEY"] = config.VLLM_API_KEY
    os.environ["OPENAI_API_BASE"] = config.VLLM_API_BASE
    return LiteModel(
        model_name=f"openai/{config.VLLM_MODEL}",
        generation_config=GenerativeModelConfig(temperature=0),
        additional_params={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )


def _load_ontology_dict() -> dict:
    with open(config.ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ontology() -> Ontology:
    return Ontology.from_json(_load_ontology_dict())


def build_cypher_system_instruction(ontology_dict: dict) -> str:
    """Cypher-generation system prompt that is explicit about arrow direction.

    Non-thinking models tend to reverse relationship directions; listing every
    relationship as an explicit arrow, plus a concrete example, strongly reduces
    that error (the SDK validates directions and otherwise rejects the query).
    """
    arrows = "\n".join(
        f"(:{rel['source']['label']})-[:{rel['label']}]->(:{rel['target']['label']})"
        for rel in ontology_dict.get("relations", [])
    )
    return (
        "Task: Generate an OpenCypher statement to query a graph database.\n\n"
        "The relationships are DIRECTED. The arrow MUST point exactly as declared "
        "(source -> target). NEVER reverse a relationship: to traverse in the "
        "opposite direction use `<-[...]-` and keep the node order.\n\n"
        "RELATIONSHIPS (source)-[:LABEL]->(target):\n"
        f"{arrows}\n\n"
        "Use only the entities, relationship types and properties in the ontology.\n"
        "Output ONLY a valid OpenCypher statement, enclosed in triple backticks.\n\n"
        "RULES:\n"
        "- Every variable used in RETURN, WHERE or ORDER BY MUST be bound in a "
        "MATCH clause first.\n"
        "- Never reference a variable that is not introduced in a MATCH pattern.\n\n"
        "DATA NOTES:\n"
        "- order_date and birthdate are strings in 'YYYY-MM-DD' format; compare "
        "them lexicographically (>=, <) — do NOT use date/datetime functions.\n"
        "- To filter by age in a given year, use a birth-date range, e.g. a "
        "40-year-old in 2024 has birthdate between '1984-01-01' and '1984-12-31'.\n\n"
        "Ontology:\n"
        "{ontology}\n\n"
        'Example 1 - question: "which products were sold in February 2024?"\n'
        "MATCH (o:Order)-[:HAS_ITEM]->(oi:OrderItem)-[:OF_PRODUCT]->(p:Product)\n"
        "WHERE o.order_date >= '2024-02-01' AND o.order_date < '2024-03-01'\n"
        "RETURN DISTINCT o.order_date, p.product_id, p.name, oi.quantity, oi.amount\n\n"
        'Example 2 - question: "which customers spent the most in Q2 2024?"\n'
        "MATCH (c:Customer)-[:PLACED]->(o:Order)-[:HAS_ITEM]->(oi:OrderItem)\n"
        "WHERE o.order_date >= '2024-04-01' AND o.order_date < '2024-07-01'\n"
        "RETURN c.id, c.name, SUM(oi.amount) AS total_spent\n"
        "ORDER BY total_spent DESC LIMIT 30\n"
    )


def create_knowledge_graph(model: Optional[LiteModel] = None) -> KnowledgeGraph:
    model = model or create_model()
    ontology_dict = _load_ontology_dict()
    return KnowledgeGraph(
        name=config.KG_NAME,
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=Ontology.from_json(ontology_dict),
        cypher_system_instruction=build_cypher_system_instruction(ontology_dict),
        host=config.FALKORDB_HOST,
        port=config.FALKORDB_PORT,
    )


def wait_for_falkordb(timeout: int = 90, interval: float = 2.0) -> None:
    """Block until FalkorDB accepts TCP connections on its port."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with socket.create_connection(
                (config.FALKORDB_HOST, config.FALKORDB_PORT), timeout=3
            ):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(interval)
    raise RuntimeError(
        f"FalkorDB not reachable at {config.FALKORDB_HOST}:{config.FALKORDB_PORT}: {last_err}"
    )


def generate_cypher(chat_session, question: str, retries: int = 5) -> str:
    """Generate a valid Cypher query, feeding validation errors back on retry.

    The SDK's built-in retry re-sends the same prompt at temperature 0, so it
    cannot recover from a rejected query (e.g. reversed relationship arrows).
    Here we pass the validator's error message back to the model so it can fix
    the direction explicitly.
    """
    if hasattr(chat_session, "get_cypher_query"):
        return chat_session.get_cypher_query(question)["cypher"]

    base_prompt = chat_session.cypher_prompt.format(question=question)
    cypher_chat = chat_session.cypher_chat_session
    ontology = chat_session.ontology
    graph = chat_session.graph
    last_error = ""

    for _ in range(retries):
        prompt = base_prompt
        if last_error:
            prompt += (
                "\n\nThe previous query was rejected:\n"
                f"{last_error}\n\n"
                "Fix it. Every variable used in RETURN, WHERE or ORDER BY must be "
                "bound in a MATCH clause first. Keep the relationship directions "
                "exactly as declared in the ontology (source -> target). Output "
                "only the corrected OpenCypher, in triple backticks."
            )
        response = cypher_chat.send_message(prompt)
        cypher = extract_cypher(response.text)
        if cypher:
            errors = validate_cypher(cypher, ontology)
            if errors:
                last_error = "\n".join(errors)
            else:
                # Static validation passes; try to run it to catch runtime issues
                # such as undefined variables ('oi' not defined).
                try:
                    graph.query(cypher)
                    return cypher
                except Exception as exc:
                    last_error = f"Query execution error: {exc}"
        else:
            last_error = "The response did not contain a Cypher statement."
        # Drop the failed exchange so the next attempt starts clean.
        try:
            cypher_chat.delete_last_message()
        except Exception:
            pass

    print(
        f"[generate_cypher] no valid query after {retries} attempts: {last_error}"
    )
    raise RuntimeError("Failed to generate a valid Cypher query: " + last_error)


def describe_output_nodes(result_set, header):
    """Flatten a FalkorDB result set (nodes + scalars) into (rows, DataFrame)."""
    columns = [col[1] for col in header]
    rows = []

    for row in result_set:
        node_data = {}
        scalar_data = {}

        for col_name, value in zip(columns, row):
            if hasattr(value, "properties"):
                node_data.setdefault(col_name, []).append(value.properties)
            else:
                scalar_data[col_name] = value

        if not node_data:
            rows.append(scalar_data)
            continue

        keys = list(node_data.keys())
        values_product = product(*[node_data[k] or [{}] for k in keys])

        for combo in values_product:
            record = {}
            for k, props in zip(keys, combo):
                for prop_key, prop_value in props.items():
                    record[f"{k}.{prop_key}"] = prop_value
            record.update(scalar_data)
            rows.append(record)

    return rows, pd.DataFrame(rows)
