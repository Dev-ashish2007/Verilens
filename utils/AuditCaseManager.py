"""
Audit Case Manager
Professional Case Lifecycle System for Audit Intelligence

Converts flagged transactions into structured, trackable audit cases.
Supports full case lifecycle: Open → Under Review → Escalated → Closed.

All functions operate on DataFrames and return copies — no in-place mutation.
Caller is responsible for writing results back to st.session_state.data.
"""

import pandas as pd
from datetime import datetime
from typing import Optional


# =====================================================
# CONSTANTS
# =====================================================

# Valid case status values (ordered by lifecycle stage)
VALID_STATUSES = [
    "Open",
    "Under Review",
    "Escalated",
    "Closed - Confirmed Fraud",
    "Closed - False Positive",
]

# Valid priority values
VALID_PRIORITIES = ["Low", "Medium", "High"]

# Severity → Priority mapping (driven by audit_engine output)
SEVERITY_TO_PRIORITY = {
    "High":   "High",
    "Medium": "Medium",
    "Low":    "Low",
}

# Statuses that are considered "closed"
CLOSED_STATUSES = {"Closed - Confirmed Fraud", "Closed - False Positive"}


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _now_iso() -> str:
    """Return current UTC-local time as ISO string."""
    return datetime.now().isoformat()


def _derive_priority(audit_severity: Optional[str]) -> str:
    """
    Derive case priority from audit_severity.

    Args:
        audit_severity: Value produced by audit_engine.py
                        ("High", "Medium", "Low", or None)

    Returns:
        Priority string: "High", "Medium", or "Low"
    """
    if not isinstance(audit_severity, str):
        return "Low"
    return SEVERITY_TO_PRIORITY.get(audit_severity.strip().capitalize(), "Low")


def _generate_case_id(index: int, year: Optional[int] = None) -> str:
    """
    Generate a deterministic case ID.

    Format: CASE-YYYY-NNNN
    Uses 1-based sequential counter within a single call batch.

    Args:
        index:  1-based position in the current batch
        year:   Calendar year (defaults to current year)

    Returns:
        Case ID string, e.g. "CASE-2026-0001"
    """
    y = year or datetime.now().year
    return f"CASE-{y}-{index:04d}"


# =====================================================
# CORE: CREATE CASE FROM A SINGLE TRANSACTION ROW
# =====================================================

def create_case_from_transaction(row: pd.Series, case_index: int = 1) -> dict:
    """
    Build a single audit case dictionary from a transaction row.

    Intended for use when processing one transaction at a time (e.g. in
    Alerts page or a future API endpoint). For bulk creation use
    create_audit_cases().

    Args:
        row:         A single DataFrame row (pd.Series) that has already
                     been processed by audit_engine.generate_audit_flags().
                     Must contain 'audit_severity'. 'transaction_id' is
                     used in the case ID if present.
        case_index:  1-based integer used to generate the case ID.

    Returns:
        Dictionary with all case fields populated. Does NOT modify the
        original DataFrame — caller must merge this back.

    Example:
        >>> for idx, row in flagged_df.iterrows():
        ...     case = create_case_from_transaction(row, idx + 1)
        ...     df.at[idx, 'audit_case_id'] = case['audit_case_id']
    """
    audit_severity = row.get("audit_severity", "Low")
    priority = _derive_priority(audit_severity)
    now = _now_iso()

    return {
        "audit_case_id":      _generate_case_id(case_index),
        "audit_status":       "Open",
        "case_priority":      priority,
        "assigned_to":        None,
        "auditor_comment":    "",
        "resolution":         "Pending",
        "resolution_notes":   "",
        "case_created_at":    now,
        "case_updated_at":    now,
    }


# =====================================================
# CORE: BULK CASE CREATION
# =====================================================

def create_audit_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create audit cases for High and Medium severity transactions.

    Initialises all case columns on every row (None for non-case rows),
    then populates case fields only for High/Medium severity rows.

    Preserves original DataFrame — operates on a copy.

    Args:
        df: DataFrame that has been processed by
            audit_engine.generate_audit_flags(). Must contain
            'audit_severity'. Silently returns with empty case columns
            if 'audit_severity' is absent.

    Returns:
        DataFrame with the following columns added:
            audit_case_id      str | None
            audit_status       str | None
            case_priority      str | None
            assigned_to        str | None
            auditor_comment    str
            resolution         str | None
            resolution_notes   str
            case_created_at    str | None
            case_updated_at    str | None

    Notes:
        - Only "High" and "Medium" severity rows receive a case ID.
        - Case IDs are sequential within this call (CASE-YYYY-0001 …).
          They are not globally unique across multiple uploads. The
          database layer (insert_audit_cases) should enforce uniqueness.
        - audit_flags may be a list or semicolon-delimited string
          (produced by audit_engine); this function does not depend on
          its type.
    """
    df_cases = df.copy()

    # Initialise all case columns with safe defaults
    df_cases["audit_case_id"]    = None
    df_cases["audit_status"]     = None
    df_cases["case_priority"]    = None
    df_cases["assigned_to"]      = None
    df_cases["auditor_comment"]  = ""
    df_cases["resolution"]       = None
    df_cases["resolution_notes"] = ""
    df_cases["case_created_at"]  = None
    df_cases["case_updated_at"]  = None

    # Guard: audit_severity must exist (produced by audit_engine)
    if "audit_severity" not in df_cases.columns:
        return df_cases

    current_year = datetime.now().year
    now = _now_iso()

    # Select rows that require a case (High and Medium only)
    flagged_mask    = df_cases["audit_severity"].isin(["High", "Medium"])
    flagged_indices = df_cases[flagged_mask].index

    for counter, row_idx in enumerate(flagged_indices, start=1):
        severity = df_cases.at[row_idx, "audit_severity"]
        priority = _derive_priority(severity)
        case_id  = _generate_case_id(counter, current_year)

        df_cases.at[row_idx, "audit_case_id"]    = case_id
        df_cases.at[row_idx, "audit_status"]     = "Open"
        df_cases.at[row_idx, "case_priority"]    = priority
        df_cases.at[row_idx, "assigned_to"]      = None
        df_cases.at[row_idx, "auditor_comment"]  = ""
        df_cases.at[row_idx, "resolution"]       = "Pending"
        df_cases.at[row_idx, "resolution_notes"] = ""
        df_cases.at[row_idx, "case_created_at"]  = now
        df_cases.at[row_idx, "case_updated_at"]  = now

    return df_cases


# =====================================================
# LIFECYCLE: UPDATE STATUS
# =====================================================

def update_case_status(
    df: pd.DataFrame,
    case_id: str,
    new_status: str,
    resolution_notes: str = "",
) -> pd.DataFrame:
    """
    Update the status of an audit case.

    Automatically updates 'case_updated_at'. If the new status is a
    closed state, sets 'resolution' to 'Resolved' and stores any
    provided resolution_notes.

    Auto-transition rule:
        "Open" → assigned → automatically becomes "Under Review"
        (handled in assign_case; update_case_status does not enforce this)

    Args:
        df:               DataFrame with audit case columns.
        case_id:          The audit_case_id to update.
        new_status:       Must be one of VALID_STATUSES.
        resolution_notes: Optional text recorded when closing a case.

    Returns:
        Updated DataFrame copy. Original is not modified.

    Raises:
        ValueError: If new_status is not in VALID_STATUSES.

    Example:
        >>> df = update_case_status(df, "CASE-2026-0001",
        ...                         "Closed - Confirmed Fraud",
        ...                         "Vendor invoice confirmed duplicate.")
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. "
            f"Must be one of: {VALID_STATUSES}"
        )

    if "audit_case_id" not in df.columns:
        return df

    df_updated = df.copy()
    mask = df_updated["audit_case_id"] == case_id

    if not mask.any():
        # Case ID not found — return unchanged
        return df_updated

    now = _now_iso()
    df_updated.loc[mask, "audit_status"]    = new_status
    df_updated.loc[mask, "case_updated_at"] = now

    if new_status in CLOSED_STATUSES:
        df_updated.loc[mask, "resolution"] = "Resolved"
        if resolution_notes:
            df_updated.loc[mask, "resolution_notes"] = resolution_notes

    return df_updated


# =====================================================
# LIFECYCLE: ASSIGN CASE
# =====================================================

def assign_case(
    df: pd.DataFrame,
    case_id: str,
    user_id: str,
) -> pd.DataFrame:
    """
    Assign an audit case to a user (by user ID or username string).

    Auto-transition: if the case is currently "Open", it moves to
    "Under Review" upon assignment.

    Args:
        df:       DataFrame with audit case columns.
        case_id:  The audit_case_id to assign.
        user_id:  User identifier (string or int). Stored as-is in
                  'assigned_to'.

    Returns:
        Updated DataFrame copy. Original is not modified.

    Example:
        >>> df = assign_case(df, "CASE-2026-0001", "auditor_jane")
    """
    if "audit_case_id" not in df.columns:
        return df

    df_updated = df.copy()
    mask = df_updated["audit_case_id"] == case_id

    if not mask.any():
        return df_updated

    now = _now_iso()
    df_updated.loc[mask, "assigned_to"]      = str(user_id)
    df_updated.loc[mask, "case_updated_at"]  = now

    # Auto-transition Open → Under Review on first assignment
    current_statuses = df_updated.loc[mask, "audit_status"]
    open_sub_mask = mask & (df_updated["audit_status"] == "Open")
    if open_sub_mask.any():
        df_updated.loc[open_sub_mask, "audit_status"] = "Under Review"

    return df_updated


# =====================================================
# LIFECYCLE: ADD COMMENT
# =====================================================

def add_case_comment(
    df: pd.DataFrame,
    case_id: str,
    comment: str,
) -> pd.DataFrame:
    """
    Append a timestamped comment to an audit case.

    Comments are appended to 'auditor_comment' with a timestamp prefix.
    Multiple comments accumulate as newline-separated entries.

    Args:
        df:       DataFrame with audit case columns.
        case_id:  The audit_case_id to comment on.
        comment:  Comment text to append.

    Returns:
        Updated DataFrame copy.

    Example:
        >>> df = add_case_comment(df, "CASE-2026-0001",
        ...                       "Requested invoice from Finance.")
    """
    if "audit_case_id" not in df.columns or not comment:
        return df

    df_updated = df.copy()
    mask = df_updated["audit_case_id"] == case_id

    if not mask.any():
        return df_updated

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_entry = f"[{timestamp}] {comment}"

    # Append to existing comment (may be empty string or prior entries)
    existing = df_updated.loc[mask, "auditor_comment"].iloc[0]
    if existing:
        updated_comment = f"{existing}\n{new_entry}"
    else:
        updated_comment = new_entry

    df_updated.loc[mask, "auditor_comment"]  = updated_comment
    df_updated.loc[mask, "case_updated_at"]  = _now_iso()

    return df_updated


# =====================================================
# QUERY: GET CASES BY STATUS
# =====================================================

def get_cases_by_status(df: pd.DataFrame, status: str) -> pd.DataFrame:
    """
    Return all rows whose audit case matches the given status.

    Args:
        df:     DataFrame with audit case columns.
        status: One of VALID_STATUSES.

    Returns:
        Filtered DataFrame copy containing only matching rows.
        Returns empty DataFrame if column is missing or no matches.

    Raises:
        ValueError: If status is not in VALID_STATUSES.

    Example:
        >>> open_cases = get_cases_by_status(df, "Open")
        >>> escalated  = get_cases_by_status(df, "Escalated")
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Must be one of: {VALID_STATUSES}"
        )

    if "audit_status" not in df.columns:
        return pd.DataFrame()

    return df[df["audit_status"] == status].copy()


# =====================================================
# QUERY: CASE SUMMARY
# =====================================================

def get_case_summary(df: pd.DataFrame) -> dict:
    """
    Return summary statistics of audit cases across all statuses.

    Counts only rows with a non-null audit_case_id (i.e. actual cases).

    Args:
        df: DataFrame with audit case columns.

    Returns:
        Dictionary with keys:
            total_cases            int
            open_cases             int
            under_review           int
            escalated              int
            closed_confirmed_fraud int
            closed_false_positive  int
            closed_cases           int   (sum of both closed types)
            high_priority          int
            medium_priority        int
            low_priority           int
    """
    empty = {
        "total_cases":            0,
        "open_cases":             0,
        "under_review":           0,
        "escalated":              0,
        "closed_confirmed_fraud": 0,
        "closed_false_positive":  0,
        "closed_cases":           0,
        "high_priority":          0,
        "medium_priority":        0,
        "low_priority":           0,
    }

    if "audit_case_id" not in df.columns:
        return empty

    cases_df = df[df["audit_case_id"].notna()].copy()

    if cases_df.empty:
        return empty

    status_counts   = cases_df["audit_status"].value_counts() if "audit_status" in cases_df.columns else {}
    priority_counts = cases_df["case_priority"].value_counts() if "case_priority" in cases_df.columns else {}

    closed_fraud    = int(status_counts.get("Closed - Confirmed Fraud", 0))
    closed_fp       = int(status_counts.get("Closed - False Positive",  0))

    return {
        "total_cases":            len(cases_df),
        "open_cases":             int(status_counts.get("Open",          0)),
        "under_review":           int(status_counts.get("Under Review",  0)),
        "escalated":              int(status_counts.get("Escalated",     0)),
        "closed_confirmed_fraud": closed_fraud,
        "closed_false_positive":  closed_fp,
        "closed_cases":           closed_fraud + closed_fp,
        "high_priority":          int(priority_counts.get("High",        0)),
        "medium_priority":        int(priority_counts.get("Medium",      0)),
        "low_priority":           int(priority_counts.get("Low",         0)),
    }


# =====================================================
# QUERY: GET CASES BY PRIORITY
# =====================================================

def get_cases_by_priority(df: pd.DataFrame, priority: str) -> pd.DataFrame:
    """
    Return all rows whose audit case matches the given priority.

    Args:
        df:       DataFrame with audit case columns.
        priority: One of "High", "Medium", "Low".

    Returns:
        Filtered DataFrame copy. Empty DataFrame if no matches.

    Raises:
        ValueError: If priority is not in VALID_PRIORITIES.
    """
    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Invalid priority '{priority}'. "
            f"Must be one of: {VALID_PRIORITIES}"
        )

    if "case_priority" not in df.columns:
        return pd.DataFrame()

    return df[df["case_priority"] == priority].copy()


# =====================================================
# MODULE INFORMATION
# =====================================================

def get_case_lifecycle_info() -> dict:
    """
    Return metadata about the case lifecycle for documentation or UI display.

    Returns:
        Dictionary describing valid statuses, priorities, and transitions.
    """
    return {
        "version":        "2.0.0",
        "valid_statuses": VALID_STATUSES,
        "valid_priorities": VALID_PRIORITIES,
        "closed_statuses": list(CLOSED_STATUSES),
        "auto_transitions": {
            "Open → Under Review": "Triggered automatically when assign_case() is called",
        },
        "priority_mapping": SEVERITY_TO_PRIORITY,
        "case_columns": [
            "audit_case_id",
            "audit_status",
            "case_priority",
            "assigned_to",
            "auditor_comment",
            "resolution",
            "resolution_notes",
            "case_created_at",
            "case_updated_at",
        ],
    }


# =====================================================
# EXAMPLE USAGE (FOR TESTING)
# =====================================================

if __name__ == "__main__":
    print("Audit Case Manager v2.0 — Case Lifecycle System")
    info = get_case_lifecycle_info()
    print(f"\nValid Statuses:   {info['valid_statuses']}")
    print(f"Valid Priorities: {info['valid_priorities']}")
    print(f"Auto Transitions: {info['auto_transitions']}")
    print(f"Priority Mapping: {info['priority_mapping']}")
    print(f"\nCase columns added to DataFrame:")
    for col in info["case_columns"]:
        print(f"  - {col}")
