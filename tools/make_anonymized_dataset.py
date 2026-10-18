#!/usr/bin/env python3
"""Build the anonymized, reduced dataset shipped with this project.

The raw source files are not part of the repository. This script documents how
the shipped `data/store_*.parquet` files were produced and can be re-run when the
raw data is available locally.

What it does:
  * keeps only orders whose product exists in the catalog,
  * samples a subset of orders,
  * replaces every personal field with synthetic values:
      - account_org_id  -> sequential integer
      - name            -> synthetic "SURNAME FIRSTNAME"
      - mobile          -> synthetic 10-digit number
      - birthdate       -> synthetic date (1940-2005)
      - business_entity_id / circuit_code -> single constant (1)
    (email is null in the source and stays null)
  * keeps the product catalog (non-personal business data).

Usage:
    python tools/make_anonymized_dataset.py \
        --orders-src raw_orders.parquet \
        --products-src raw_products.parquet \
        --out-dir data --n-orders 10000 --seed 42
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import polars as pl

FIRST_NAMES = [
    "MARIO", "LUIGI", "ANNA", "MARIA", "GIUSEPPE", "FRANCESCO", "CARMELA",
    "ANTONIO", "GIULIA", "PAOLO", "SARA", "LUCA", "ELENA", "DAVIDE", "CHIARA",
    "MATTEO", "ALESSIA", "STEFANO", "FRANCESCA", "ANDREA", "ROBERTA",
    "SALVATORE", "VALENTINA", "MARCO", "FEDERICA",
]

LAST_NAMES = [
    "ROSSI", "BIANCHI", "FERRARI", "RUSSO", "GALLO", "COLOMBO", "RICCI",
    "MARINO", "GRECO", "BRUNO", "GALLI", "CONTI", "DELUCA", "MANCINI",
    "COSTA", "GIORDANO", "RIZZO", "LOMBARDI", "MORETTI", "BARBIERI",
    "FONTANA", "CARUSO", "SANTORO", "PALUMBO", "SEVERINO",
]

BIRTH_START = date(1940, 1, 1)
BIRTH_END = date(2005, 12, 31)


def synthetic_name(index: int) -> str:
    last = LAST_NAMES[index % len(LAST_NAMES)]
    first = FIRST_NAMES[(index // len(LAST_NAMES)) % len(FIRST_NAMES)]
    return f"{last} {first}"


def synthetic_mobile(index: int) -> str:
    return f"333{(index % 10_000_000):07d}"


def synthetic_birthdate(rng: random.Random) -> str:
    span = (BIRTH_END - BIRTH_START).days
    return (BIRTH_START + timedelta(days=rng.randint(0, span))).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders-src", required=True)
    parser.add_argument("--products-src", required=True)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--n-orders", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = pl.read_parquet(args.orders_src)
    products = pl.read_parquet(args.products_src)

    # Keep only orders whose product is present in the catalog.
    orders = orders.join(
        products.select("product_id").unique(), on="product_id", how="inner"
    )

    if orders.height > args.n_orders:
        orders = orders.sample(n=args.n_orders, seed=args.seed)

    # One synthetic identity per customer.
    customer_ids = orders.select("account_org_id").unique().sort("account_org_id")[
        "account_org_id"
    ].to_list()
    rng = random.Random(args.seed)

    identities = pl.DataFrame(
        {
            "account_org_id": customer_ids,
            "new_account_org_id": list(range(1, len(customer_ids) + 1)),
            "new_name": [synthetic_name(i) for i in range(len(customer_ids))],
            "new_email": [f"customer{i:05d}@example.com" for i in range(len(customer_ids))],
            "new_mobile": [synthetic_mobile(i) for i in range(len(customer_ids))],
            "new_birthdate": [synthetic_birthdate(rng) for _ in customer_ids],
        }
    )

    orders = orders.join(identities, on="account_org_id", how="left")

    orders = orders.with_columns(
        pl.col("new_account_org_id").alias("account_org_id"),
        pl.col("new_name").alias("name"),
        pl.when(pl.col("email").is_not_null())
        .then(pl.col("new_email"))
        .otherwise(None)
        .alias("email"),
        pl.col("new_mobile").alias("mobile"),
        pl.col("new_birthdate").alias("birthdate"),
        pl.lit(1, dtype=pl.Int64).alias("business_entity_id"),
        pl.lit(1, dtype=pl.Int64).alias("circuit_code"),
    ).select(
        [
            "account_org_id",
            "orderdate",
            "business_entity_id",
            "circuit_code",
            "product_id",
            "quantity",
            "amount",
            "name",
            "email",
            "mobile",
            "birthdate",
        ]
    )

    orders.write_parquet(out_dir / "store_orders.parquet")
    products.write_parquet(out_dir / "store_products.parquet")

    print(f"orders written: {orders.height} (customers={len(customer_ids)})")
    print(f"products written: {products.height}")
    print(f"output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
