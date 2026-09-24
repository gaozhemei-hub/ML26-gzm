"""Task-aware delivery acceptance and same-session recovery.

This module does not replace TrustGuard's audits, environment ledger or hashes.
"""
from .delivery import EvidenceDelivery, EvidenceOutcome

__all__ = ["EvidenceDelivery", "EvidenceOutcome"]
