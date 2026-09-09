"""
state.py
--------
Shared LangGraph state for the Insurance Claim Processing Agent.

Note on reducers: three nodes (document verification, eligibility check,
fraud detection) run in PARALLEL as fan-out branches from the entry
point and then fan back in to a single `merge` node. Any field that
more than one parallel branch could theoretically write must use an
Annotated reducer so LangGraph can merge concurrent updates instead of
raising an "InvalidUpdateError". We use `operator.add` on the `trace`
list for this purpose; every other field is written by exactly one
node, so plain overwrite is fine.
"""

import operator
from typing import Annotated, List, Optional, TypedDict


class ClaimState(TypedDict, total=False):
    # ---- Input -------------------------------------------------------
    claim_id: str
    policy_number: str
    claimant_name: str
    claim_type: str  # "Auto" | "Health" | "Property" | "Travel"
    claim_amount: float
    policy_coverage_limit: float
    incident_date: str  # ISO date string
    policy_start_date: str
    policy_end_date: str
    prior_claims_last_year: int
    submitted_documents: List[str]

    # ---- Document Verification Agent ---------------------------------
    required_documents: List[str]
    missing_documents: List[str]
    documents_verified: bool
    document_notes: str

    # ---- Eligibility Check Agent --------------------------------------
    eligibility_status: bool
    eligibility_notes: str

    # ---- Fraud Detection Agent -----------------------------------------
    fraud_risk_score: int  # 0-100
    fraud_flags: List[str]
    fraud_notes: str

    # ---- Claim Summary Agent -------------------------------------------
    claim_summary: str

    # ---- Decision / Routing --------------------------------------------
    decision: str  # "auto_approve" | "reject" | "human_review"
    decision_reason: str

    # ---- Human Approval Agent (human-in-the-loop) -----------------------
    human_decision: Optional[str]  # "approved" | "rejected" | None (pending)
    human_notes: Optional[str]

    # ---- Final outcome ---------------------------------------------------
    final_status: str  # "Approved" | "Rejected" | "Awaiting Human Review"

    # ---- Execution trace (parallel-safe via operator.add) ----------------
    trace: Annotated[List[str], operator.add]
