from typing import Any

import pandas as pd


def load_support_data() -> pd.DataFrame:
    """Load sample customer support data for analysis."""

    data: list[dict[str, Any]] = [
        {
            "month": "2026-01",
            "tickets": 4200,
            "avg_resolution_hours": 18.2,
            "staff_count": 52,
        },
        {
            "month": "2026-02",
            "tickets": 4500,
            "avg_resolution_hours": 19.1,
            "staff_count": 52,
        },
        {
            "month": "2026-03",
            "tickets": 5100,
            "avg_resolution_hours": 22.4,
            "staff_count": 50,
        },
        {
            "month": "2026-04",
            "tickets": 5700,
            "avg_resolution_hours": 26.8,
            "staff_count": 48,
        },
    ]

    return pd.DataFrame(data)