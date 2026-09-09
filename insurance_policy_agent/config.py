"""
config.py
---------
Static business rules that a real insurer would keep in a database.
Kept simple/hard-coded here for demo purposes.
"""

REQUIRED_DOCUMENTS = {
    "Auto": ["claim_form", "police_report", "photos_of_damage", "repair_estimate"],
    "Health": ["claim_form", "medical_bill", "doctor_report"],
    "Property": ["claim_form", "photos_of_damage", "repair_estimate", "ownership_proof"],
    "Travel": ["claim_form", "boarding_pass", "receipt"],
}

# Claim types each policy plan covers (demo simplification: one plan "Standard")
COVERED_CLAIM_TYPES = {"Auto", "Health", "Property", "Travel"}

# Thresholds
FRAUD_ESCALATION_THRESHOLD = 60      # >= this score -> escalate for human review
FRAUD_AUTO_REJECT_THRESHOLD = 90     # >= this score -> auto reject (extremely high confidence fraud)
HIGH_VALUE_CLAIM_THRESHOLD = 15000   # claims at/above this amount always go to human review
RECENT_POLICY_WINDOW_DAYS = 14       # incident within N days of policy start looks suspicious
