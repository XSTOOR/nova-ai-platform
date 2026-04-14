"""
NOVA AI Platform — Audit Logger
=================================
Structured audit logging for all MCP tool invocations.

Features:
  - Every tool call logged with timestamp, tool_name, input, output, latency
  - Structured JSON log entries
  - Query/filter by tool, date range, success/failure
  - Summary statistics (call counts, avg latency, error rates)
  - In-memory storage with optional file persistence
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Audit Log Entry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """A single audit log entry for a tool invocation."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tool_name: str = ""
    input_params: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    session_id: Optional[str] = None
    customer_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"AuditEntry({status} {self.tool_name}, {self.latency_ms:.1f}ms)"


# ──────────────────────────────────────────────────────────────────────
# Audit Logger
# ──────────────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Structured audit logger for MCP tool invocations.

    Usage:
        audit = AuditLogger()
        with audit.track("order_lookup", {"order_id": "ORD-123"}) as tracker:
            result = lookup_order("ORD-123")
            tracker.set_output(result)
    """

    def __init__(self, persist_dir: Optional[str] = None, session_id: Optional[str] = None):
        """
        Initialize the audit logger.

        Args:
            persist_dir: Directory to persist log files. If None, in-memory only.
            session_id: Optional session ID to group related calls.
        """
        self._entries: list[AuditEntry] = []
        self._session_id = session_id or str(uuid.uuid4())
        self._persist_dir = Path(persist_dir) if persist_dir else None

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    def track(self, tool_name: str, input_params: dict,
              customer_id: Optional[str] = None) -> "AuditTracker":
        """
        Create an audit tracker for a tool invocation.

        Args:
            tool_name: Name of the tool being called.
            input_params: Input parameters to the tool.
            customer_id: Optional customer ID for the session.

        Returns:
            AuditTracker context manager that auto-logs on exit.
        """
        return AuditTracker(self, tool_name, input_params, customer_id)

    def log_entry(self, entry: AuditEntry) -> None:
        """Log an audit entry."""
        entry.session_id = self._session_id
        self._entries.append(entry)

        # Persist to file if configured
        if self._persist_dir:
            log_file = self._persist_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(entry.to_json() + "\n")

        status = "SUCCESS" if entry.success else "FAILURE"
        logger.info(
            f"AUDIT | {status} | {entry.tool_name} | "
            f"{entry.latency_ms:.1f}ms | session={entry.session_id[:8]}..."
        )

    @property
    def entries(self) -> list[AuditEntry]:
        """Get all audit entries."""
        return self._entries.copy()

    def get_entries(
        self,
        tool_name: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Query audit entries with optional filters.

        Args:
            tool_name: Filter by tool name.
            success: Filter by success status.
            limit: Maximum entries to return.

        Returns:
            Filtered list of audit entries.
        """
        results = self._entries
        if tool_name:
            results = [e for e in results if e.tool_name == tool_name]
        if success is not None:
            results = [e for e in results if e.success == success]
        return results[-limit:]

    def get_summary(self) -> dict:
        """
        Get summary statistics of all logged calls.

        Returns:
            Dict with call counts, latency stats, error rates per tool.
        """
        if not self._entries:
            return {"total_calls": 0}

        total = len(self._entries)
        successes = sum(1 for e in self._entries if e.success)
        failures = total - successes
        latencies = [e.latency_ms for e in self._entries]

        # Per-tool breakdown
        tool_stats: dict[str, dict] = {}
        for entry in self._entries:
            if entry.tool_name not in tool_stats:
                tool_stats[entry.tool_name] = {
                    "calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_latency_ms": 0.0,
                }
            stats = tool_stats[entry.tool_name]
            stats["calls"] += 1
            stats["total_latency_ms"] += entry.latency_ms
            if entry.success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1

        # Calculate averages
        for name, stats in tool_stats.items():
            stats["avg_latency_ms"] = round(
                stats["total_latency_ms"] / stats["calls"], 2
            )
            stats["success_rate"] = round(
                stats["successes"] / stats["calls"] * 100, 1
            )

        return {
            "total_calls": total,
            "successes": successes,
            "failures": failures,
            "success_rate": round(successes / total * 100, 1),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "tools": tool_stats,
            "session_id": self._session_id,
        }

    def get_audit_trail(self) -> list[dict]:
        """Get the full audit trail as a list of dicts."""
        return [e.to_dict() for e in self._entries]

    def clear(self) -> None:
        """Clear all audit entries."""
        self._entries.clear()


# ──────────────────────────────────────────────────────────────────────
# Audit Tracker (Context Manager)
# ──────────────────────────────────────────────────────────────────────

class AuditTracker:
    """
    Context manager that tracks tool execution and auto-logs.

    Usage:
        with audit.track("order_lookup", {"order_id": "ORD-123"}) as t:
            result = do_something()
            t.set_output(result)
            # Auto-logged when exiting the 'with' block
    """

    def __init__(
        self,
        audit_logger: AuditLogger,
        tool_name: str,
        input_params: dict,
        customer_id: Optional[str] = None,
    ):
        self._audit_logger = audit_logger
        self._entry = AuditEntry(
            tool_name=tool_name,
            input_params=input_params,
            customer_id=customer_id,
        )
        self._start_time: float = 0.0

    def __enter__(self) -> "AuditTracker":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = (time.perf_counter() - self._start_time) * 1000  # ms
        self._entry.latency_ms = round(elapsed, 2)

        if exc_type:
            self._entry.success = False
            self._entry.error = f"{exc_type.__name__}: {exc_val}"

        self._audit_logger.log_entry(self._entry)

    def set_output(self, output: dict) -> None:
        """Set the output result for the tracked call."""
        self._entry.output = output

    def set_error(self, error: str) -> None:
        """Manually set an error message."""
        self._entry.success = False
        self._entry.error = error

    def set_customer_id(self, customer_id: str) -> None:
        """Associate a customer ID with this call."""
        self._entry.customer_id = customer_id