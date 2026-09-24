"""Task-aware delivery acceptance and same-session recovery.

This module does not replace TrustGuard's audits, environment ledger or hashes.
"""
from .delivery import EvidenceDelivery, EvidenceOutcome
from tabularx.schema.evidence import EvidenceAttempt, EvidenceManifest, EvidenceRecord

__all__ = ["EvidenceDelivery", "EvidenceOutcome", "EvidenceAttempt", "EvidenceRecord", "EvidenceManifest"]
