"""
llm.py
------
Thin LLM wrapper used by every agent node.

Supports three providers, chosen at runtime (from the Streamlit sidebar,
or via environment variables for non-UI use):

  - "mock"      : no API key needed. Deterministic rule-flavoured
                   reasoning so the whole graph runs end-to-end for free.
  - "openai"    : OpenAI models (e.g. gpt-4o-mini), via langchain_openai.
  - "deepseek"  : DeepSeek models (e.g. deepseek-chat). DeepSeek exposes
                   an OpenAI-compatible /chat/completions endpoint, so we
                   reuse langchain_openai.ChatOpenAI and just point it at
                   DeepSeek's base_url.

The key is NEVER written to disk or baked into the code -- it's held
only in memory for the current process/session (see `configure()` and
how streamlit_app.py calls it from a password-masked sidebar field).

Every node calls `call_llm(system_prompt, user_prompt)` and gets back a
string, regardless of which provider is active, so nothing in nodes.py
needs to change when you switch providers.
"""

import os
import json

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
}

# Current runtime configuration (mutated by configure(), read by call_llm()).
_state = {
    "provider": "mock",
    "api_key": None,
    "model": None,
    "client": None,
}


def configure(provider: str, api_key: str | None = None, model: str | None = None):
    """
    Set which LLM provider/key/model subsequent call_llm() calls should use.
    Called once per Streamlit run (or once at startup for non-UI use).

    provider: "mock" | "openai" | "deepseek"
    api_key:  the user's own API key (required for "openai"/"deepseek")
    model:    optional override, otherwise a sensible default per provider
    """
    provider = (provider or "mock").lower()
    _state["provider"] = provider
    _state["api_key"] = api_key
    _state["client"] = None  # rebuilt lazily below

    if provider == "mock" or not api_key:
        _state["provider"] = "mock"
        return

    _state["model"] = model or DEFAULT_MODELS.get(provider, "gpt-4o-mini")

    try:
        from langchain_openai import ChatOpenAI

        kwargs = dict(model=_state["model"], temperature=0, api_key=api_key)
        if provider == "deepseek":
            kwargs["base_url"] = DEEPSEEK_BASE_URL

        _state["client"] = ChatOpenAI(**kwargs)
    except Exception as e:  # pragma: no cover
        print(f"[llm.py] Could not initialize {provider} client, falling back to mock: {e}")
        _state["provider"] = "mock"
        _state["client"] = None


def current_provider() -> str:
    return _state["provider"]


def is_live() -> bool:
    return _state["provider"] != "mock" and _state["client"] is not None


# ---------------------------------------------------------------------
# Bootstrap from environment variables so the graph also works outside
# Streamlit (e.g. scripts, tests) without any explicit configure() call.
# ---------------------------------------------------------------------
if os.environ.get("DEEPSEEK_API_KEY"):
    configure("deepseek", os.environ["DEEPSEEK_API_KEY"], os.environ.get("CLAIM_AGENT_MODEL"))
elif os.environ.get("OPENAI_API_KEY"):
    configure("openai", os.environ["OPENAI_API_KEY"], os.environ.get("CLAIM_AGENT_MODEL"))


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the configured LLM and return raw text content."""
    if is_live():
        from langchain_core.messages import SystemMessage, HumanMessage

        try:
            resp = _state["client"].invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return resp.content
        except Exception as e:
            # Surface the real error (bad key, quota, network) rather than
            # silently mock-answering, so the user knows to check their key.
            raise RuntimeError(
                f"{_state['provider']} API call failed: {e}. "
                "Check your API key/quota, or switch to Mock mode in the sidebar."
            ) from e

    # ---- Mock fallback -----------------------------------------------
    # Deterministic, rule-flavoured "reasoning" so the demo works without
    # any API key. Each node passes enough context in user_prompt that we
    # can produce a plausible structured answer.
    return _mock_reasoning(system_prompt, user_prompt)


def _mock_reasoning(system_prompt: str, user_prompt: str) -> str:
    """
    Very small heuristic engine that mimics what an LLM would return for
    each of our node types, based on keywords baked into the prompts by
    the calling node. Returns JSON text in all cases, matching what the
    real LLM is instructed to return.
    """
    if "DOCUMENT_VERIFICATION_TASK" in system_prompt:
        missing = "MISSING_DOCS::" in user_prompt and "MISSING_DOCS::none" not in user_prompt
        if missing:
            return json.dumps(
                {
                    "verified": False,
                    "notes": "One or more required documents were not found in the submission. "
                    "The claim cannot proceed to eligibility review until they are provided.",
                }
            )
        return json.dumps(
            {
                "verified": True,
                "notes": "All required documents for this claim type are present and appear "
                "consistent with the claim details provided.",
            }
        )

    if "ELIGIBILITY_TASK" in system_prompt:
        if "POLICY_EXPIRED::true" in user_prompt:
            return json.dumps(
                {
                    "eligible": False,
                    "notes": "The policy was not active on the incident date -- the policy period "
                    "had already ended before the loss occurred.",
                }
            )
        if "COVERAGE_MISMATCH::true" in user_prompt:
            return json.dumps(
                {
                    "eligible": False,
                    "notes": "The claim type is not covered under the policy's current plan.",
                }
            )
        return json.dumps(
            {
                "eligible": True,
                "notes": "The policy was active on the date of the incident and the claim type "
                "falls within the covered categories.",
            }
        )

    if "FRAUD_TASK" in system_prompt:
        score = 10
        flags = []
        if "AMOUNT_RATIO_HIGH::true" in user_prompt:
            score += 45
            flags.append("Claim amount is unusually high relative to policy coverage limits")
        if "RECENT_POLICY_START::true" in user_prompt:
            score += 30
            flags.append("Incident occurred shortly after the policy was purchased")
        if "ROUND_NUMBER_AMOUNT::true" in user_prompt:
            score += 10
            flags.append("Claim amount is a suspiciously round figure")
        if "PRIOR_CLAIMS_HIGH::true" in user_prompt:
            score += 15
            flags.append("Claimant has an elevated recent claims history")
        score = min(score, 97)
        if not flags:
            flags.append("No significant fraud indicators detected")
        return json.dumps({"fraud_risk_score": score, "flags": flags})

    if "SUMMARY_TASK" in system_prompt:
        return json.dumps(
            {
                "summary": "Mock summary generated without a live LLM connection. In production "
                "this would be a concise, underwriter-ready narrative combining document "
                "status, eligibility findings, and fraud risk into a recommendation."
            }
        )

    return json.dumps({"note": "mock response"})
