from typing import Any

import pandas as pd

from multi_agent_ops.state import OpsState
from multi_agent_ops.tools.data_tools import load_support_data


def calculate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate key metrics from customer support data."""

    first_row = df.iloc[0]
    last_row = df.iloc[-1]

    ticket_change_pct = (
        (last_row["tickets"] - first_row["tickets"])
        / first_row["tickets"]
    ) * 100

    resolution_time_change_pct = (
        (last_row["avg_resolution_hours"]
         - first_row["avg_resolution_hours"])
        / first_row["avg_resolution_hours"]
    ) * 100

    staff_change_pct = (
        (last_row["staff_count"] - first_row["staff_count"])
        / first_row["staff_count"]
    ) * 100

    tickets_per_staff_first = (
        first_row["tickets"] / first_row["staff_count"]
    )

    tickets_per_staff_last = (
        last_row["tickets"] / last_row["staff_count"]
    )

    workload_change_pct = (
        (tickets_per_staff_last - tickets_per_staff_first)
        / tickets_per_staff_first
    ) * 100

    return {
        "ticket_change_pct": round(ticket_change_pct, 2),
        "resolution_time_change_pct": round(resolution_time_change_pct, 2),
        "staff_change_pct": round(staff_change_pct, 2),
        "tickets_per_staff_first": round(tickets_per_staff_first, 2),
        "tickets_per_staff_last": round(tickets_per_staff_last, 2),
        "workload_change_pct": round(workload_change_pct, 2),
    }


def data_analysis_agent(state: OpsState) -> dict:
    """Analyze customer support data and identify measurable trends."""

    df = load_support_data()

    metrics = calculate_metrics(df)

    findings = {
        "metrics": metrics,
        "data": df.to_dict(orient="records"),
    }

    return {
        "data_findings": findings,
        "status": "DATA_ANALYSIS_COMPLETED",
    }