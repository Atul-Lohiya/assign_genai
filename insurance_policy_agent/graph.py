"""
graph.py
--------
Wires the five agents into a LangGraph StateGraph:

                        ┌──────────────────────────┐
                        │           START          │
                        └─────────────┬────────────┘
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
   Document Verification      Eligibility Check          Fraud Detection      <- run in PARALLEL
              └────────────────────────┼─────────────────────────┘
                                       ▼
                                    merge
                                       ▼
                              Claim Summary Agent
                                       ▼
                               Decision Router
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
              auto_approve        auto_reject     Human Approval Agent
                     │                │           (interrupt / HITL)
                     ▼                ▼                 ▼
                    END              END               END

A checkpointer (MemorySaver) is required for the interrupt()-based
human-in-the-loop step to work -- it lets the graph pause after
`human_approval_node` calls interrupt() and resume later from the same
point once the human's decision is supplied via Command(resume=...).
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import ClaimState
from nodes import (
    document_verification_node,
    eligibility_check_node,
    fraud_detection_node,
    merge_node,
    claim_summary_node,
    decision_node,
    route_after_decision,
    human_approval_node,
    auto_approve_node,
    auto_reject_node,
)


def build_graph():
    graph = StateGraph(ClaimState)

    # Nodes
    graph.add_node("document_verification", document_verification_node)
    graph.add_node("eligibility_check", eligibility_check_node)
    graph.add_node("fraud_detection", fraud_detection_node)
    graph.add_node("merge", merge_node)
    graph.add_node("claim_summary", claim_summary_node)
    graph.add_node("decision", decision_node)
    graph.add_node("auto_approve", auto_approve_node)
    graph.add_node("auto_reject", auto_reject_node)
    graph.add_node("human_approval", human_approval_node)

    # --- Parallel fan-out from START ---------------------------------
    graph.add_edge(START, "document_verification")
    graph.add_edge(START, "eligibility_check")
    graph.add_edge(START, "fraud_detection")

    # --- Fan-in: all three parallel branches must complete before merge
    graph.add_edge("document_verification", "merge")
    graph.add_edge("eligibility_check", "merge")
    graph.add_edge("fraud_detection", "merge")

    # --- Sequential from here -----------------------------------------
    graph.add_edge("merge", "claim_summary")
    graph.add_edge("claim_summary", "decision")

    # --- Conditional routing based on decision -------------------------
    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "auto_approve": "auto_approve",
            "reject": "auto_reject",
            "human_review": "human_approval",
        },
    )

    graph.add_edge("auto_approve", END)
    graph.add_edge("auto_reject", END)
    graph.add_edge("human_approval", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


# A module-level singleton so Streamlit (which reruns the script on every
# interaction) doesn't rebuild the graph unnecessarily.
claim_graph = build_graph()
