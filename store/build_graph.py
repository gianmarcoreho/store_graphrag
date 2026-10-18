"""Phase 1 — Build and populate the Store knowledge graph in FalkorDB.

Reads the parquet input files from ``data/``, creates indexes, and loads
Products, Brands, Categories, Customers, Stores, Circuits, Orders and OrderItems
together with their relationships.

Usage:
    python -m store.build_graph
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import polars as pl

from . import config
from .kg import create_knowledge_graph, wait_for_falkordb


def _normalize_value(v: Any) -> Any:
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _create_indexes(kg) -> None:
    print("Creating indexes...")
    for label, attr in [
        ("Customer", "id"),
        ("Store", "id"),
        ("Order", "order_id"),
        ("OrderItem", "id"),
        ("Product", "product_id"),
        ("Brand", "name"),
        ("Category", "name"),
        ("Circuit", "code"),
    ]:
        kg.graph.query(f"CREATE INDEX ON :{label}({attr})")


def _load_products(kg) -> None:
    df_products = pl.read_parquet(config.PRODUCTS_PARQUET)
    df_products = df_products.with_columns(
        pl.col("Description Extended").fill_null("")
    )
    df_products_clean = df_products.rename(
        {
            "Nome": "name",
            "Description Extended": "description",
            "EAN": "ean",
            "MINSAN": "minsan",
            "Prezzo di listino": "list_price",
            "Brand": "brand",
            "Category": "category",
        }
    )

    print("Inserting products...")
    product_query = """
    UNWIND $rows AS row
    MERGE (p:Product {product_id: row.product_id})
    SET
        p.name = row.name,
        p.description = row.description,
        p.ean = row.ean,
        p.minsan = row.minsan,
        p.list_price = row.list_price
    """
    for chunk in df_products_clean.iter_slices(n_rows=config.BATCH_SIZE):
        rows = chunk.select(
            ["product_id", "name", "description", "ean", "minsan", "list_price"]
        ).to_dicts()
        kg.graph.query(product_query, params={"rows": rows})

    print("Inserting brands...")
    brand_query = "UNWIND $rows AS row MERGE (b:Brand {name: row.brand})"
    df_brand = df_products_clean.select("brand").drop_nulls().unique()
    for chunk in df_brand.iter_slices(n_rows=config.BATCH_SIZE):
        kg.graph.query(brand_query, params={"rows": chunk.to_dicts()})

    print("Inserting categories...")
    cat_query = "UNWIND $rows AS row MERGE (c:Category {name: row.category})"
    df_cat = df_products_clean.select("category").drop_nulls().unique()
    for chunk in df_cat.iter_slices(n_rows=config.BATCH_SIZE):
        kg.graph.query(cat_query, params={"rows": chunk.to_dicts()})

    print("Linking products -> brands / categories...")
    rel_query = """
    UNWIND $rows AS row
    MATCH (p:Product {product_id: row.product_id})
    MATCH (b:Brand {name: row.brand})
    MATCH (c:Category {name: row.category})
    MERGE (p)-[:BELONGS_TO]->(b)
    MERGE (p)-[:IN_CATEGORY]->(c)
    """
    df_products_rel = (
        df_products_clean.select(["product_id", "brand", "category"]).drop_nulls()
    )
    for chunk in df_products_rel.iter_slices(n_rows=config.BATCH_SIZE):
        kg.graph.query(rel_query, params={"rows": chunk.to_dicts()})


def _load_orders(kg) -> None:
    df_orders = pl.read_parquet(config.ORDERS_PARQUET)
    df_orders = df_orders.with_row_index("row_id")
    df_orders = df_orders.with_columns(
        [
            (
                pl.col("account_org_id").cast(pl.String)
                + "_"
                + pl.col("orderdate").cast(pl.String)
            ).alias("order_id"),
            pl.col("row_id").cast(pl.String).alias("order_item_id"),
            pl.col("orderdate").str.to_datetime(strict=False).dt.date().alias("order_date"),
        ]
    )

    print("Inserting orders + order items...")
    order_query = """
    UNWIND $rows AS row
    MERGE (cust:Customer {id: row.account_org_id})
    SET
        cust.name = row.name,
        cust.email = row.email,
        cust.mobile = row.mobile,
        cust.birthdate = row.birthdate
    MERGE (store:Store {id: row.business_entity_id})
    MERGE (circuit:Circuit {code: row.circuit_code})
    MERGE (ord:Order {order_id: row.order_id})
    SET
        ord.order_date = row.order_date
    MERGE (cust)-[:PLACED]->(ord)
    MERGE (ord)-[:FROM_STORE]->(store)
    MERGE (ord)-[:VIA_CIRCUIT]->(circuit)
    MERGE (item:OrderItem {id: row.order_item_id})
    SET
        item.quantity = row.quantity,
        item.amount = row.amount
    MERGE (ord)-[:HAS_ITEM]->(item)
    WITH row, item
    MATCH (p:Product {product_id: row.product_id})
    MERGE (item)-[:OF_PRODUCT]->(p)
    """
    for chunk in df_orders.iter_slices(n_rows=config.BATCH_SIZE):
        rows = [
            {k: _normalize_value(v) for k, v in r.items()}
            for r in chunk.to_dicts()
        ]
        kg.graph.query(order_query, params={"rows": rows})


def build() -> None:
    wait_for_falkordb()
    kg = create_knowledge_graph()
    _create_indexes(kg)
    _load_products(kg)
    _load_orders(kg)
    print("Graph successfully populated.")


if __name__ == "__main__":
    build()
