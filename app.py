"""Streamlit UI: ask a natural-language question and query the Store graph.

This file lives at the project root on purpose: Streamlit runs it as a
top-level script, so it imports the package with absolute imports.
"""

from __future__ import annotations

import json
import time

import streamlit as st

from store.kg import (
    create_knowledge_graph,
    describe_output_nodes,
    generate_cypher,
    wait_for_falkordb,
)


@st.cache_resource
def load_graph():
    wait_for_falkordb()
    return create_knowledge_graph()


st.set_page_config(page_title="Store GraphRAG", layout="wide")
st.title("Store Orders & Products — Knowledge Graph")
st.caption(
    "Ask a question in natural language. The app turns it into a Cypher query "
    "for FalkorDB and shows the matching rows."
)

kg = load_graph()
chat_kg = kg.chat_session()

with st.form("query_form"):
    query_user = st.text_input(
        "Your question",
        placeholder="e.g. which products were sold in February 2025?",
    )
    run_query = st.form_submit_button("Run query")

if run_query and query_user:
    start_time = time.perf_counter()
    try:
        with st.spinner("Generating the Cypher query..."):
            cypher_query = generate_cypher(chat_kg, query_user)

        with st.spinner("Querying the knowledge graph..."):
            res_out = kg.graph.query(cypher_query)
            out_list, out_df = describe_output_nodes(res_out.result_set, res_out.header)
    except Exception:
        st.warning("no-answer")
        st.stop()

    execution_time = round(time.perf_counter() - start_time, 3)

    st.success(f"{len(out_df)} rows returned")
    st.info(f"⏱ {execution_time} s")

    with st.expander("🔎 Generated Cypher query"):
        st.code(cypher_query, language="cypher")

    st.subheader("Results")
    if len(out_df) > 0:
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download results (JSON)",
                data=json.dumps(out_list, indent=2, default=str),
                file_name="query_results.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                label="⬇️ Download results (CSV)",
                data=out_df.to_csv(index=False),
                file_name="query_results.csv",
                mime="text/csv",
            )
    else:
        st.warning("No results found.")
