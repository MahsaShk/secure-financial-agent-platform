from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from secure_financial_agent.agent import SecureFinancialAgent
from secure_financial_agent.audit import InMemoryAuditSink
from secure_financial_agent.ingest import load_chunks
from secure_financial_agent.retrieve import TfidfRetriever
from secure_financial_agent.schemas import AgentRequest

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def build_agent() -> tuple[SecureFinancialAgent, InMemoryAuditSink]:
    chunks = load_chunks(ROOT / "data" / "sample_documents.json")
    audit_sink = InMemoryAuditSink()
    return SecureFinancialAgent(TfidfRetriever(chunks), audit_sink=audit_sink), audit_sink


st.set_page_config(page_title="Secure Financial Agent MVP", page_icon="🔐")
st.title("Secure Financial Agent MVP")
st.caption(
    "Local synthetic data for engineering review only. "
    "Outputs are educational and are not financial advice."
)

agent, audit_sink = build_agent()
question = st.text_input("Question", "What was ACME revenue in 2025?")

if st.button("Ask", type="primary"):
    try:
        response = agent.answer(AgentRequest(question=question))
    except ValueError as error:
        st.error(str(error))
    else:
        st.json(response.to_dict())
        with st.expander("Latest audit event"):
            st.code(json.dumps(audit_sink.events[-1].to_dict(), indent=2), language="json")

