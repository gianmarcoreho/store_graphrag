"""Central configuration for the Store GraphRAG project.

Every setting is read from environment variables (with sensible defaults), so
the same code runs both locally and inside Docker. A local ``.env`` file is
loaded automatically when present.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (one level above this file).
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load a local .env if present (silently ignored when missing, e.g. in Docker).
load_dotenv(dotenv_path=BASE_DIR / ".env")

# --- FalkorDB -------------------------------------------------------------
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
KG_NAME = os.getenv("KG_NAME", "store_orders_products_new")

# --- LLM (Qwen served by vLLM, OpenAI-compatible) -------------------------
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "your-model-name")
# The vLLM endpoint is keyless, but LiteLLM requires an OpenAI key to be set.
VLLM_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# --- Input data files ------------------------------------------------------
ONTOLOGY_PATH = DATA_DIR / "store_orders_ontology_new.json"
ORDERS_PARQUET = DATA_DIR / "store_orders.parquet"
PRODUCTS_PARQUET = DATA_DIR / "store_products.parquet"

# --- Data loading ----------------------------------------------------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))
