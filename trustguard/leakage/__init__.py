"""Standalone leakage-risk extension for TabularX.

This directory intentionally has no changes to the existing ``tabularx``
package.  It can be copied, reviewed, or merged as one isolated unit.
"""

from .audit import LeakageAuditor, AuditFinding, AuditResult
from .config import LeakageAuditConfig
from .reporting import write_audit_artifacts
from .selection import validate_selection_protocol
from .safe_model_search import SafeModelRun, SafeSearchResult, safe_search_models

__all__ = [
    "LeakageAuditor", "LeakageAuditConfig", "AuditFinding", "AuditResult",
    "write_audit_artifacts", "validate_selection_protocol",
    "SafeModelRun", "SafeSearchResult", "safe_search_models",
]
