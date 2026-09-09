"""
streamlit_app.py
-----------------
Streamlit front-end for the Insurance Claim Processing Agent.

Run with:
    streamlit run streamlit_app.py
"""

import uuid
import random
import streamlit as st
from langgraph.types import Command
from graph import claim_graph
from config import REQUIRED_DOCUMENTS
import llm

print("Starting app")
print(st.session_state)

st.set_page_config(page_title="Insurance Claim Processing Agent", page_icon="📋", layout="wide")

# -----------------------------------------------------------------------
# Demo scenarios (matches the 5 example test cases from the brief)
# -----------------------------------------------------------------------
SCENARIOS = {
    "1. Complete claim, valid documents -> Auto Approve": dict(
        claim_id="CLM-1001",
        policy_number="POL-9001",
        claimant_name="Asha Rao",
        claim_type="Auto",
        claim_amount=2200.0,
        policy_coverage_limit=50000.0,
        incident_date="2026-05-10",
        policy_start_date="2024-01-01",
        policy_end_date="2027-01-01",
        prior_claims_last_year=0,
        submitted_documents=["claim_form", "police_report", "photos_of_damage", "repair_estimate"],
    ),
    "2. Missing required documents -> Reject": dict(
        claim_id="CLM-1002",
        policy_number="POL-9002",
        claimant_name="Daniel Kim",
        claim_type="Property",
        claim_amount=4500.0,
        policy_coverage_limit=80000.0,
        incident_date="2026-04-02",
        policy_start_date="2023-06-01",
        policy_end_date="2027-06-01",
        prior_claims_last_year=0,
        submitted_documents=["claim_form", "photos_of_damage"],  # missing estimate + ownership proof
    ),
    "3. Suspicious claim amount -> Human Review": dict(
        claim_id="CLM-1003",
        policy_number="POL-9003",
        claimant_name="Priya Menon",
        claim_type="Auto",
        claim_amount=46850.0,
        policy_coverage_limit=50000.0,  # ratio 0.94 -> fraud flag
        incident_date="2026-06-01",
        policy_start_date="2026-05-20",  # 12 days before incident -> recent policy flag
        policy_end_date="2027-05-20",
        prior_claims_last_year=1,
        submitted_documents=["claim_form", "police_report", "photos_of_damage", "repair_estimate"],
    ),
    "4. Expired policy -> Reject": dict(
        claim_id="CLM-1004",
        policy_number="POL-9004",
        claimant_name="Wei Zhang",
        claim_type="Health",
        claim_amount=1800.0,
        policy_coverage_limit=20000.0,
        incident_date="2026-07-01",
        policy_start_date="2023-01-01",
        policy_end_date="2025-12-31",  # policy ended before incident
        prior_claims_last_year=1,
        submitted_documents=["claim_form", "medical_bill", "doctor_report"],
    ),
    "5. High-value claim, valid documents -> Human Review": dict(
        claim_id="CLM-1005",
        policy_number="POL-9005",
        claimant_name="Grace Odhiambo",
        claim_type="Property",
        claim_amount=32000.0,  # above HIGH_VALUE_CLAIM_THRESHOLD
        policy_coverage_limit=100000.0,
        incident_date="2026-03-15",
        policy_start_date="2022-01-01",
        policy_end_date="2027-01-01",
        prior_claims_last_year=0,
        submitted_documents=["claim_form", "photos_of_damage", "repair_estimate", "ownership_proof"],
    ),
}

def sync_claim_to_widgets(claim_data):
    st.session_state["claim_id"] = claim_data["claim_id"]
    st.session_state["policy_number"] = claim_data["policy_number"]
    st.session_state["claimant_name"] = claim_data["claimant_name"]
    st.session_state["claim_type"] = claim_data["claim_type"]
    st.session_state["claim_amount"] = claim_data["claim_amount"]
    st.session_state["policy_coverage_limit"] = claim_data["policy_coverage_limit"]
    st.session_state["incident_date"] = claim_data["incident_date"]
    st.session_state["policy_start_date"] = claim_data["policy_start_date"]
    st.session_state["policy_end_date"] = claim_data["policy_end_date"]
    st.session_state["prior_claims_last_year"] = claim_data["prior_claims_last_year"]
    st.session_state["submitted_documents"] = claim_data["submitted_documents"]

# -----------------------------------------------------------------------
# Run the graph
# -----------------------------------------------------------------------
def process_claim(claim_data: dict):
    thread_id = str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = claim_graph.invoke(claim_data, config=config)
    except RuntimeError as e:
        st.session_state.result_state = None
        st.session_state.llm_error = str(e)
        return

    st.session_state.llm_error = None
    st.session_state.result_state = result

    if "__interrupt__" in result:
        st.session_state.awaiting_human = True
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.awaiting_human = False
        st.session_state.interrupt_payload = None


def resume_with_human_decision(decision: str, notes: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        result = claim_graph.invoke(
            Command(resume={"decision": decision, "notes": notes}), config=config
        )
    except RuntimeError as e:
        st.session_state.llm_error = str(e)
        return
    st.session_state.llm_error = None
    st.session_state.result_state = result
    st.session_state.awaiting_human = False
    st.session_state.interrupt_payload = None

def submit_claim():
    print("submit click handler executed")
    st.session_state.claim_details_expander = False

    claim_data = {
                "claim_id": st.session_state.claim_id,
                "policy_number": st.session_state.policy_number,
                "claimant_name": st.session_state.claimant_name,
                "claim_type": st.session_state.claim_type,
                "claim_amount": st.session_state.claim_amount,
                "policy_coverage_limit": st.session_state.policy_coverage_limit,
                "incident_date": st.session_state.incident_date,
                "policy_start_date": st.session_state.policy_start_date,
                "policy_end_date": st.session_state.policy_end_date,
                "prior_claims_last_year": st.session_state.prior_claims_last_year,
                "submitted_documents": st.session_state.submitted_documents,
            }
    
    print("Processing claim")
    process_claim(claim_data)
    print("Claim processing completed")

    # Store submitted claim
    st.session_state.claim_data = claim_data

    # Optional: store a flag that results are available
    st.session_state.claim_submitted = True

def reset_assessment():
    print("Resetting assessment")
    st.session_state.claim_submitted = False
    st.session_state.claim_data = {}


CLAIM_TYPES = list(REQUIRED_DOCUMENTS.keys())

# -----------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "result_state" not in st.session_state:
    st.session_state.result_state = None
if "awaiting_human" not in st.session_state:
    st.session_state.awaiting_human = False
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None


st.title("Insurance Claim Processing Agent")
st.caption("Built with LangGraph - parallel verification/eligibility/fraud checks, conditional routing, and human-in-the-loop escalation.")

scenario_name = None
claim_data = None

# -----------------------------------------------------------------------
# Sidebar: LLM provider + API key (never written to disk -- kept only in
# this browser session's memory and passed straight to the API call).
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("LLM Settings")
    provider_label = st.selectbox(
        "Provider",
        ["Mock (rule based, no key needed)", "DeepSeek", "OpenAI"],
        help="Pick which model powers the 4 reasoning agents (Document Verification, "
        "Eligibility, Fraud Detection, Claim Summary). The Decision router itself is "
        "rule-based and doesn't call the LLM.",
    )

    api_key = None
    model_override = None
    provider_key = "mock"

    if provider_label == "DeepSeek":
        provider_key = "deepseek"
        api_key = st.text_input(
            "DeepSeek API key",
            type="password",
            placeholder="sk-...",
            help="Get one at platform.deepseek.com. Stored only in memory for this session.",
        )
        model_override = st.text_input("Model", value="deepseek-chat")
    elif provider_label == "OpenAI":
        provider_key = "openai"
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Stored only in memory for this session.",
        )
        model_override = st.text_input("Model", value="gpt-4o-mini")

    llm.configure(provider_key, api_key or None, model_override or None)

    if llm.is_live():
        st.success(f"Connected - using **{llm.current_provider()}** for live reasoning.", icon="✅")
    else:
        st.info(
            "Running in **mock mode** - no key entered, so the 4 reasoning agents use "
            "built-in rule-based logic instead of a live LLM call. Change the mode and enter a key above to "
            "switch to real reasoning.",
            icon="ℹ️",
        )

    st.divider()

    st.header("Claim Scenario")
    mode = st.radio("Select a mode", ["Preloaded demo scenario", "Custom claim"], key="claim_mode", on_change=reset_assessment)

    if mode == "Preloaded demo scenario":
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), on_change=reset_assessment)


# ---------------------------------------------------------
# Initialize / reset sample claim data for selected scenario
# ---------------------------------------------------------

if mode == "Preloaded demo scenario":
    claim_data = dict(SCENARIOS[scenario_name])
else:
    # Auto-generate identifiers for custom claim
    claim_data = dict(
        claim_id=f"CLM-{random.randint(0, 9999):04d}",
        policy_number=f"POL-{random.randint(0, 9999):04d}",
        claimant_name="",
        claim_type="Auto",
        claim_amount=0.0,
        policy_coverage_limit=0.0,
        incident_date="",
        policy_start_date="",
        policy_end_date="",
        prior_claims_last_year=0,
        submitted_documents=[],
    )

if (
    "claim_scenario" not in st.session_state
    or st.session_state.claim_scenario != scenario_name
):
    st.session_state.claim_data = claim_data
    st.session_state.claim_scenario = scenario_name
    sync_claim_to_widgets(claim_data)


# ---------------------------------------------------------
# Claim Input Section
# ---------------------------------------------------------

with st.expander(
    "Claim Details",
    expanded=not st.session_state.get("claim_submitted", False),
    key="claim_details_expander",
    on_change = "rerun"
    
):

    # Row 1
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        claim_id = st.text_input(
            "Claim ID",
            key="claim_id"
        )

    with col2:
        policy_number = st.text_input(
            "Policy Number",
            key="policy_number"
        )

    with col3:
        claimant_name = st.text_input(
            "Claimant Name",
            key="claimant_name"
        )

    with col4:
        claim_type = st.selectbox(
            "Claim Type",
            CLAIM_TYPES,
            key="claim_type"
        )


    # Row 2
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        claim_amount = st.number_input(
            "Claim Amount ($)",
            min_value=0.0,
            step=100.0,
            key="claim_amount"
        )

    with col2:
        policy_coverage_limit = st.number_input(
            "Policy Coverage Limit ($)",
            min_value=0.0,
            step=1000.0,
            key="policy_coverage_limit"
        )

    with col3:
        incident_date = st.text_input(
            "Incident Date (YYYY-MM-DD)",
            key="incident_date"
        )

    with col4:
        policy_start_date = st.text_input(
            "Policy Start Date (YYYY-MM-DD)",
            key="policy_start_date"
        )


    # Row 3
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        policy_end_date = st.text_input(
            "Policy End Date (YYYY-MM-DD)",
            key="policy_end_date"
        )

    with col2:
        prior_claims_last_year = st.number_input(
            "Prior Claims Last Year",
            min_value=0,
            step=1,
            key="prior_claims_last_year"
        )

    # Determine documents based on selected claim type
    available_docs = REQUIRED_DOCUMENTS.get(claim_type, [])

    with col3:
        submitted_documents = st.multiselect(
            "Submitted Documents",
            options=available_docs,
            key="submitted_documents"
        )

    # Fourth column intentionally left empty
    with col4:
        st.empty()


# -----------------------------------------------------
# Submit button
# -----------------------------------------------------
submit_clicked = st.button(
    "Submit Claim",
    type="primary",
    use_container_width=True,
    on_click=submit_claim
)


# ---------------------------------------------------------
# Process submission
# ---------------------------------------------------------

if "llm_error" not in st.session_state:
    st.session_state.llm_error = None

if st.session_state.llm_error:
    st.error(f"{st.session_state.llm_error}")
    
# ---------------------------------------------------------
# Results Section - Always visible
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("Claim Assessment")

    if st.session_state.get("claim_submitted", False):

        claim_data = st.session_state.claim_data

        # -----------------------------------------------------------------------
        # Display results
        # -----------------------------------------------------------------------
        result = st.session_state.result_state

        if result:
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Document Verification")
                st.write("Verified" if result.get("documents_verified") else "Not Verified")
                if result.get("missing_documents"):
                    st.write(f"Missing: {', '.join(result['missing_documents'])}")
                st.caption(result.get("document_notes", ""))

            with col2:
                st.subheader("Eligibility Check")
                st.write("Eligible" if result.get("eligibility_status") else "Not Eligible")
                st.caption(result.get("eligibility_notes", ""))

            with col3:
                st.subheader("Fraud Detection")
                score = result.get("fraud_risk_score", 0)
                st.metric("Fraud Risk Score", f"{score}/100")
                for flag in result.get("fraud_flags", []):
                    st.caption(f"{flag}")

            st.divider()
            st.subheader("Claim Summary")
            st.write(result.get("claim_summary", ""))

            st.divider()

            if st.session_state.awaiting_human and st.session_state.interrupt_payload:
                payload = st.session_state.interrupt_payload
                st.warning(
                    f"**Human review required** - {payload.get('message', '')}\n\n"
                    f"Reason for escalation: {result.get('decision_reason', '')}"
                )
                with st.form("human_review_form"):
                    notes = st.text_area("Reviewer notes")
                    c1, c2 = st.columns(2)
                    approve = c1.form_submit_button("Approve Claim", use_container_width=True)
                    reject = c2.form_submit_button("Reject Claim", use_container_width=True)

                if approve:
                    resume_with_human_decision("approved", notes)
                    st.rerun()
                if reject:
                    resume_with_human_decision("rejected", notes)
                    st.rerun()

            else:
                final_status = result.get("final_status", "Pending")
                decision_reason = result.get("decision_reason", "")
                if final_status == "Approved":
                    st.success(f"### Final Status: {final_status}")
                elif final_status == "Rejected":
                    st.error(f"### Final Status: {final_status}")
                else:
                    st.info(f"### Final Status: {final_status}")
                st.caption(f"Decision reasoning: {decision_reason}")
                if result.get("human_decision"):
                    st.caption(
                        f"Human reviewer decision: **{result['human_decision']}** - notes: "
                        f"{result.get('human_notes') or '(none)'}"
                    )

            with st.expander("Full agent execution trace"):
                for line in result.get("trace", []):
                    st.text(line)

            with st.expander("Raw final state (debug)"):
                st.json({k: v for k, v in result.items() if k != "__interrupt__"})

        else:
            st.info("Choose a scenario or build a custom claim in the sidebar, then click **Run Claim Through Agent**.")

    else:
        st.info("Submit a claim to view the assessment.")

print("Ending app")
st.session_state.rerun = False