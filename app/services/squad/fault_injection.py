"""Week 10: Controlled HTTP 500 fault injection for the version/deprecation worker.

The fault is only active when explicitly enabled for the experiment. It affects
the version/deprecation worker call for exactly one designated evaluation case,
returning an HTTP 500 error that the orchestrator must handle gracefully.

The fault-injection mechanism is isolated from normal production operation:
it is disabled by default and only enabled via an explicit environment variable
or experiment flag.
"""

import os
from typing import Optional

# Environment variable to enable fault injection for the experiment.
# Format: "gh-006" (case ID) - only that case gets the 500 from the version worker.
FAULT_INJECTION_CASE_ID = os.environ.get("RAGGER_FAULT_CASE_ID", "")

# When set to "1", the version/deprecation worker raises an HTTP 500 for the
# designated case. Default "0" means fault injection is disabled.
FAULT_INJECTION_ENABLED = os.environ.get("RAGGER_FAULT_INJECTION", "0") == "1"


def is_fault_enabled() -> bool:
    """Return True if controlled fault injection is active."""
    return FAULT_INJECTION_ENABLED and bool(FAULT_INJECTION_CASE_ID)


def should_inject_fault(case_id: str, worker_name: str = "version_deprecation") -> bool:
    """Return True if the given case/worker should receive an injected HTTP 500.

    Only the version/deprecation worker is affected, and only for the
    designated case ID. Never affects normal production requests unless the
    experiment flag is explicitly set.
    """
    if not is_fault_enabled():
        return False
    if worker_name != "version_deprecation":
        return False
    return case_id == FAULT_INJECTION_CASE_ID


class InjectedHTTP500Error(Exception):
    """Raised by the version/deprecation worker when fault injection is active."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        super().__init__(f"Injected HTTP 500 for case {case_id} (version/deprecation worker)")
