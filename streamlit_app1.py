# app.py
"""
Streamlit App for Product Review RAG System
Improved version with reliable LLM query analysis via Groq
"""

import streamlit as st
import os
import json
import numpy as np
from typing import List, Tuple, Optional
from rapidfuzz import fuzz, process, utils

# Your custom classes
from product_review_rag import ProductReviewRAG
from product_review_qa import ProductReviewQA, ReviewChunk, LLMIntegrations

# ────────────────────────────────────────────────────────────────
# Page Configuration
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Review RAG Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stButton > button {
        font-weight: bold;
    }
    .result-card {
        background-color: #0f1117;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4e8cff;
    }
    .answer-card {
        background-color: #0f1117;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #a0d0ff;
    }
    .score-badge {
        background-color: #28a745;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    hr {
        margin: 1.8rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# Title & Description
# ────────────────────────────────────────────────────────────────
st.title("🔍 Product Review RAG Explorer")
st.markdown("Ask questions about products based on real customer reviews. Powered by semantic search & LLM reasoning.")

# ────────────────────────────────────────────────────────────────
# Sidebar Settings
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    storage_dir = st.text_input("Storage directory", value="./rag_storage")
    default_top_k = st.slider("Default number of results", 1, 10, 4)
    st.markdown("---")
    st.info("Examples:\n• Is the cable durable?\n• Compare Wayona and Ambrane\n• Best fast charging cable")

# ────────────────────────────────────────────────────────────────
# Load RAG system (cached)
# ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_rag_system(dir_path: str):
    if not os.path.exists(dir_path):
        st.error(f"Storage directory not found: **{dir_path}**\nRun your indexing script first.")
        st.stop()
    
    with st.spinner("Loading embeddings and chunks..."):
        rag = ProductReviewRAG()
        try:
            rag.load_locally(save_dir=dir_path)
            return rag
        except Exception as e:
            st.error(f"Load failed: {str(e)}")
            st.stop()

rag = load_rag_system(storage_dir)

# Cache QA system
@st.cache_resource
def get_qa(_rag):
    return ProductReviewQA(_rag)

qa = get_qa(rag)

st.success(f"Loaded **{len(rag.chunks)}** review chunks.", icon="✅")

# ────────────────────────────────────────────────────────────────
# Main Interface
# ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Your question:",
        placeholder="Is the cable durable? Compare Wayona and Ambrane? Which is best?",
        key="query_input"
    )

with col2:
    product_filter = st.text_input(
        "Product filter (optional)",
        placeholder="Wayona, boAt, Ambrane, B07JW9H4J1",
        help="Partial name, brand or product ID",
        key="product_filter"
    ).strip() or None

top_k = st.slider(
    "Number of review chunks to show",
    1, 10, default_top_k, key="top_k_slider"
)

# ────────────────────────────────────────────────────────────────
# Search Logic
# ────────────────────────────────────────────────────────────────
if st.button("🔎 Search Reviews", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing query..."):
            try:
                # ── 1. Analyze query with LLM (Groq) ──────────────────────
                analyze_prompt = f"""
You must respond with **valid JSON only**. No explanations, no markdown, no code blocks, no extra text.

User question: "{query}"

Return exactly:

{{
  "is_comparison": true or false,
  "extracted_products": ["product1", "product2"] or []
}}

Rules:
- is_comparison = true ONLY if question compares or mentions multiple products
- Use double quotes
- No trailing commas
- No ```json or other formatting
"""

                llm = LLMIntegrations()
                analysis_raw = llm.call_llm_groq(
                    prompt=analyze_prompt,
                    model="llama-3.3-70b-versatile"
                )

                # ── Clean response ────────────────────────────────────────
                cleaned = analysis_raw.strip()

                # Remove markdown fences
                if cleaned.startswith("```json"):
                    cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                elif cleaned.startswith("```"):
                    cleaned = cleaned.split("```", 2)[1].strip() if len(cleaned.split("```")) > 2 else cleaned

                # Extract JSON object
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                if start >= 0 and end > start:
                    cleaned = cleaned[start:end]

                # Parse
                try:
                    analysis = json.loads(cleaned)
                    is_comparison = analysis.get("is_comparison", False)
                    extracted_products = analysis.get("extracted_products", [])
                except json.JSONDecodeError as je:
                    st.warning("Could not parse LLM JSON response")
                    with st.expander("Raw LLM output (debug)"):
                        st.code(analysis_raw, language="text")
                    is_comparison = False
                    extracted_products = []
                except Exception as e:
                    st.error(f"Unexpected error in analysis: {e}")
                    is_comparison = False
                    extracted_products = []

                # ── 2. Prioritize user-provided filter ────────────────────
                if product_filter:
                    extracted_products = list(set([product_filter] + extracted_products))

                # ── 3. Match extracted products to real ones ─────────────
                matched_products = []  # list of (real_name, score)

                for prod in extracted_products:
                    filter_lower = utils.default_process(prod.strip())
                    product_scores = {}

                    for chunk in rag.chunks:
                        name_clean = utils.default_process(chunk.product_name)
                        scores = [
                            fuzz.partial_ratio(filter_lower, name_clean),
                            fuzz.token_sort_ratio(filter_lower, name_clean),
                            fuzz.partial_token_sort_ratio(filter_lower, name_clean),
                        ]
                        best_score = max(scores)
                        id_score = fuzz.partial_ratio(filter_lower, chunk.product_id.lower())
                        best_score = max(best_score, id_score * 0.9)

                        product_scores[chunk.product_name] = max(
                            product_scores.get(chunk.product_name, 0), best_score
                        )

                    if product_scores:
                        best_name = max(product_scores, key=product_scores.get)
                        best_score = product_scores[best_name]
                        if best_score >= 72:
                            matched_products.append((best_name, best_score))

                # Sort by score descending
                matched_products.sort(key=lambda x: x[1], reverse=True)

                if matched_products:
                    st.markdown(
                        f"**Matched products:** {', '.join([f'{n} ({s:.0f}%)' for n, s in matched_products])}"
                    )

                     # ── 4. Retrieve chunks ────────────────────────────────────
                    all_results = []
                    filter_used = ", ".join([n for n, _ in matched_products]) if matched_products else product_filter

                    if matched_products:
                        k_per_product = max(1, top_k // len(matched_products))
                        for prod_name, _ in matched_products:
                            prod_results = rag.retrieve(
                                query=query,
                                product_filter=prod_name,
                                top_k=k_per_product + 1
                            )
                            all_results.extend(prod_results)
                        # Re-sort combined results
                        all_results.sort(key=lambda x: x[1], reverse=True)
                        results = all_results[:top_k]
                    else:
                        results = rag.retrieve(
                            query=query,
                            product_filter=product_filter,
                            top_k=top_k
                        )

                    # ── 5. Generate answer ────────────────────────────────────
                    answer = qa.answer_question(
                        question=query,
                        product_filter=filter_used,
                        top_k=min(4 * max(1, len(matched_products)), top_k * 2)
                    )

                    # ── Display results ───────────────────────────────────────
                    st.markdown("### Answer")
                    with st.container(border=True):
                        st.markdown(answer)

                    st.markdown("---")

                    if not results:
                        st.info("No relevant reviews found.")
                    else:
                        st.markdown(f"### Top {len(results)} relevant reviews")
                        for i, (chunk, score) in enumerate(results, 1):
                            with st.expander(
                                f"#{i} • {score:.0%} • {chunk.product_name[:60]}...",
                                expanded=(i <= 2)
                            ):
                                st.markdown(f"**Type:** {chunk.chunk_type}")
                                if chunk.user_name and chunk.user_name != "Aggregated":
                                    st.markdown(f"**User:** {chunk.user_name}")
                                if chunk.review_title:
                                    st.markdown(f"**Title:** {chunk.review_title}")
                                st.markdown("**Content:**")
                                st.markdown(f"```text\n{chunk.chunk_text.strip()}\n```")

                elif product_filter:
                    st.warning("No strong match found for the provided filter. Using fallback.")
                else:
                    st.info("No specific products detected in query.")

               
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("Product Review RAG • Streamlit • February 2026")