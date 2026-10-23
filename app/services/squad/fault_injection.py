"""Week 10: Controlled HTTP 500 fault injection for the version/deprecation worker.

The fault is only active when explicitly enabled for the experiment. It affects
the version/deprecation worker call for exactly one designated evaluation case,
returning an HTTP 500 error that the orchestrator must handle gracefully.

The fault-injection mechanism is isolated from normal production operation:
it is disabled by default and only enabled via an explicit environment variable
or experiment flag.
"""

import os
from contextlib import contextmanager
from typing import Iterator

# Fault injection is read from the environment at call time (not import time) so
# an experiment can enable it for exactly one scoped run and the benchmark that
# runs before/after it is never affected.
#   RAGGER_FAULT_INJECTION=1   enable (default "0" = disabled)
#   RAGGER_FAULT_CASE_ID=gh-006  the single case whose version worker gets the 500
ENV_ENABLED = "RAGGER_FAULT_INJECTION"
ENV_CASE_ID = "RAGGER_FAULT_CASE_ID"


def fault_case_id() -> str:
    return os.environ.get(ENV_CASE_ID, "")


def is_fault_enabled() -> bool:
    """Return True if controlled fault injection is active."""
    return os.environ.get(ENV_ENABLED, "0") == "1" and bool(fault_case_id())


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
    return case_id == fault_case_id()


@contextmanager
def inject_fault_for(case_id: str) -> Iterator[None]:
    """Enable the HTTP 500 for `case_id` only inside this block."""
    previous = {k: os.environ.get(k) for k in (ENV_ENABLED, ENV_CASE_ID)}
    os.environ[ENV_ENABLED] = "1"
    os.environ[ENV_CASE_ID] = case_id
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class InjectedHTTP500Error(Exception):
    """Raised by the version/deprecation worker when fault injection is active."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        super().__init__(f"Injected HTTP 500 for case {case_id} (version/deprecation worker)")
