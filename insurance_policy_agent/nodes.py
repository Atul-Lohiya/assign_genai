"""
nodes.py
--------
The five required agents, implemented as LangGraph nodes:

  1. document_verification_node
  2. eligibility_check_node
  3. fraud_detection_node
  4. claim_summary_node
  5. human_approval_node

Plus two small support nodes:
  - merge_node        (fan-in point after the 3 parallel checks)
  - decision_node      (rule-based router that reads the 3 parallel results)

Nodes 1-3 are fanned out in parallel from the graph's entry point (see
graph.py) and fan back in to `merge_node` before summary/decision.
"""

import json
from datetime import datetime

from langgraph.types import interrupt

from state import ClaimState
from config import (
    REQUIRED_DOCUMENTS,
    COVERED_CLAIM_TYPES,
    FRAUD_ESCALATION_THRESHOLD,
    FRAUD_AUTO_REJECT_THRESHOLD,
    HIGH_VALUE_CLAIM_THRESHOLD,
    RECENT_POLICY_WINDOW_DAYS,
)
from llm import call_llm


def _parse_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d")


# ---------------------------------------------------------------------
# 1. Document Verification Agent
# ---------------------------------------------------------------------
def document_verification_node(state: ClaimState) -> dict:
    claim_type = state["claim_type"]
    required = REQUIRED_DOCUMENTS.get(claim_type, ["claim_form"])
    submitted = set(state.get("submitted_documents", []))
    missing = [d for d in required if d not in submitted]

    system_prompt = (
        "You are the Document Verification Agent for an insurance claims system. "
        "DOCUMENT_VERIFICATION_TASK. Decide if all required documents are present, "
        "given the required list and the submitted list. Respond ONLY as JSON with "
        "keys 'verified' (bool) and 'notes' (string)."
    )
    user_prompt = (
        f"Claim type: {claim_type}\n"
        f"Required documents: {required}\n"
        f"Submitted documents: {sorted(submitted)}\n"
        f"MISSING_DOCS::{'none' if not missing else ','.join(missing)}"
    )

    raw_llm_response = call_llm(system_prompt, user_prompt)
    try:
        parsed = json.loads(raw_llm_response)
    except Exception:
        parsed = {"verified": not missing, "notes": raw_llm_response}

    verified = bool(parsed.get("verified", not missing)) and not missing
    notes = parsed.get("notes", "")

    return {
        "required_documents": required,
        "missing_documents": missing,
        "documents_verified": verified,
        "document_notes": notes,
        "trace": [
            f"[Document Verification Agent] verified={verified}, missing={missing or 'none'}"
        ],
    }


# ---------------------------------------------------------------------
# 2. Eligibility Check Agent
# ---------------------------------------------------------------------
def eligibility_check_node(state: ClaimState) -> dict:
    incident_date = _parse_date(state["incident_date"])
    policy_start_date = _parse_date(state["policy_start_date"])
    policy_end_date = _parse_date(state["policy_end_date"])

    is_policy_expired = not (policy_start_date <= incident_date <= policy_end_date)
    coverage_mismatch = state["claim_type"] not in COVERED_CLAIM_TYPES

    system_prompt = (
        "You are the Eligibility Check Agent for an insurance claims system. "
        "ELIGIBILITY_TASK. Decide if the claim is eligible for coverage based on "
        "policy dates and claim type. Respond ONLY as JSON with keys 'eligible' (bool) "
        "and 'notes' (string)."
    )
    user_prompt = (
        f"Claim type: {state['claim_type']}\n"
        f"Incident date: {state['incident_date']}\n"
        f"Policy period: {state['policy_start_date']} to {state['policy_end_date']}\n"
        f"POLICY_EXPIRED::{'true' if is_policy_expired else 'false'}\n"
        f"COVERAGE_MISMATCH::{'true' if coverage_mismatch else 'false'}"
    )

    raw_llm_response = call_llm(system_prompt, user_prompt)
    try:
        parsed = json.loads(raw_llm_response)
    except Exception:
        parsed = {"eligible": not (is_policy_expired or coverage_mismatch), "notes": raw_llm_response}

    eligible = bool(parsed.get("eligible", True)) and not is_policy_expired and not coverage_mismatch
    notes = parsed.get("notes", "")

    return {
        "eligibility_status": eligible,
        "eligibility_notes": notes,
        "trace": [f"[Eligibility Check Agent] eligible={eligible}"],
    }


# ---------------------------------------------------------------------
# 3. Fraud Detection Agent
# ---------------------------------------------------------------------
def fraud_detection_node(state: ClaimState) -> dict:
    claim_amount = state["claim_amount"]
    coverage_limit = state.get("policy_coverage_limit", claim_amount) or claim_amount
    ratio_high = coverage_limit > 0 and (claim_amount / coverage_limit) >= 0.9
    policy_start_date = _parse_date(state["policy_start_date"])
    incident_date = _parse_date(state["incident_date"])
    recent_policy = 0 <= (incident_date - policy_start_date).days <= RECENT_POLICY_WINDOW_DAYS
    round_number = claim_amount % 1000 == 0 and claim_amount > 0
    prior_claims_high = state.get("prior_claims_last_year", 0) >= 3

    system_prompt = (
        "You are the Fraud Detection Agent for an insurance claims system. "
        "FRAUD_TASK. Estimate a fraud_risk_score from 0-100 and list any flags. "
        "Respond ONLY as JSON with keys 'fraud_risk_score' (int) and 'flags' (list of strings)."
    )
    user_prompt = (
        f"Claim amount: {claim_amount}, Policy coverage limit: {coverage_limit}\n"
        f"AMOUNT_RATIO_HIGH::{'true' if ratio_high else 'false'}\n"
        f"RECENT_POLICY_START::{'true' if recent_policy else 'false'}\n"
        f"ROUND_NUMBER_AMOUNT::{'true' if round_number else 'false'}\n"
        f"PRIOR_CLAIMS_HIGH::{'true' if prior_claims_high else 'false'}"
    )

    raw_llm_response = call_llm(system_prompt, user_prompt)
    try:
        parsed = json.loads(raw_llm_response)
    except Exception:
        parsed = {"fraud_risk_score": 50 if ratio_high or recent_policy else 10, "flags": [raw_llm_response]}

    score = int(parsed.get("fraud_risk_score", 10))
    flags = parsed.get("flags", [])

    return {
        "fraud_risk_score": score,
        "fraud_flags": flags,
        "fraud_notes": "; ".join(flags),
        "trace": [f"[Fraud Detection Agent] risk_score={score}, flags={flags}"],
    }


# ---------------------------------------------------------------------
# Fan-in merge node (no-op placeholder so we have one clean join point)
# ---------------------------------------------------------------------
def merge_node(state: ClaimState) -> dict:
    return {"trace": ["[Merge] all parallel checks complete -- proceeding to summary"]}


# ---------------------------------------------------------------------
# 4. Claim Summary Agent
# ---------------------------------------------------------------------
def claim_summary_node(state: ClaimState) -> dict:
    system_prompt = (
        "You are the Claim Summary Agent for an insurance claims system. "
        "SUMMARY_TASK. Write a concise, professional 3-5 sentence summary of the claim "
        "for an underwriter, combining document status, eligibility, and fraud risk. "
        "Respond ONLY as JSON with key 'summary' (string)."
    )
    user_prompt = (
        f"Claim ID: {state['claim_id']}\n"
        f"Claimant: {state['claimant_name']}\n"
        f"Claim type: {state['claim_type']}, Amount: {state['claim_amount']}\n"
        f"Documents verified: {state['documents_verified']} "
        f"(missing: {state.get('missing_documents') or 'none'})\n"
        f"Document notes: {state.get('document_notes', '')}\n"
        f"Eligibility: {state['eligibility_status']} -- {state.get('eligibility_notes', '')}\n"
        f"Fraud risk score: {state['fraud_risk_score']} -- flags: {state.get('fraud_flags', [])}"
    )

    raw_llm_response = call_llm(system_prompt, user_prompt)
    try:
        parsed = json.loads(raw_llm_response)
        summary = parsed.get("summary", raw_llm_response)
    except Exception:
        summary = raw_llm_response

    return {
        "claim_summary": summary,
        "trace": ["[Claim Summary Agent] summary generated"],
    }


# ---------------------------------------------------------------------
# Decision node (rule-based router -- reads outputs of the 3 parallel agents)
# ---------------------------------------------------------------------
def decision_node(state: ClaimState) -> dict:
    if not state["documents_verified"]:
        decision = "reject"
        reason = f"Missing required documents: {', '.join(state.get('missing_documents', []))}."
    elif not state["eligibility_status"]:
        decision = "reject"
        reason = state.get("eligibility_notes", "Policy is not eligible for this claim.")
    elif state["fraud_risk_score"] >= FRAUD_AUTO_REJECT_THRESHOLD:
        decision = "reject"
        reason = f"Fraud risk score of {state['fraud_risk_score']} exceeds the auto-reject threshold."
    elif (
        state["fraud_risk_score"] >= FRAUD_ESCALATION_THRESHOLD
        or state["claim_amount"] >= HIGH_VALUE_CLAIM_THRESHOLD
    ):
        decision = "human_review"
        reasons = []
        if state["fraud_risk_score"] >= FRAUD_ESCALATION_THRESHOLD:
            reasons.append(f"elevated fraud risk score ({state['fraud_risk_score']})")
        if state["claim_amount"] >= HIGH_VALUE_CLAIM_THRESHOLD:
            reasons.append(f"high claim value (${state['claim_amount']:,.2f})")
        reason = "Escalated for human review due to " + " and ".join(reasons) + "."
    else:
        decision = "auto_approve"
        reason = "All documents verified, policy eligible, and fraud risk is low."

    return {
        "decision": decision,
        "decision_reason": reason,
        "trace": [f"[Decision Router] decision={decision} -- {reason}"],
    }


def route_after_decision(state: ClaimState) -> str:
    """Conditional-edge function used by the graph to pick the next node."""
    return state["decision"]


# ---------------------------------------------------------------------
# 5. Human Approval Agent (human-in-the-loop)
# ---------------------------------------------------------------------
def human_approval_node(state: ClaimState) -> dict:
    """
    Pauses the graph using LangGraph's `interrupt()`. The Streamlit app
    resumes execution with `Command(resume={"decision": ..., "notes": ...})`
    once a human reviewer submits their verdict.
    """
    payload = interrupt(
        {
            "claim_id": state["claim_id"],
            "claimant_name": state["claimant_name"],
            "claim_summary": state.get("claim_summary", ""),
            "decision_reason": state.get("decision_reason", ""),
            "fraud_risk_score": state.get("fraud_risk_score"),
            "fraud_flags": state.get("fraud_flags", []),
            "claim_amount": state["claim_amount"],
            "message": "Human review required. Please approve or reject this claim.",
        }
    )
    human_decision = payload.get("decision", "rejected")
    human_notes = payload.get("notes", "")

    return {
        "human_decision": human_decision,
        "human_notes": human_notes,
        "final_status": "Approved" if human_decision == "approved" else "Rejected",
        "trace": [f"[Human Approval Agent] human_decision={human_decision}"],
    }


# ---------------------------------------------------------------------
# Terminal nodes for the two non-human-review outcomes
# ---------------------------------------------------------------------
def auto_approve_node(state: ClaimState) -> dict:
    return {
        "final_status": "Approved",
        "trace": ["[Auto Approve] claim approved automatically"],
    }


def auto_reject_node(state: ClaimState) -> dict:
    return {
        "final_status": "Rejected",
        "trace": ["[Auto Reject] claim rejected automatically"],
    }
