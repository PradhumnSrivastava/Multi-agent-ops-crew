from io import BytesIO

import pandas as pd


def load_support_data(file_bytes: bytes) -> pd.DataFrame:
    """Load company support data from an uploaded CSV file."""

    if not file_bytes:
        raise ValueError("Uploaded CSV file is empty.")

    df = pd.read_csv(BytesIO(file_bytes))

    if df.empty:
        raise ValueError("Uploaded CSV file does not contain any data.")

    return df


def validate_support_data(df: pd.DataFrame) -> None:
    """Validate the minimum columns required for support analysis."""

    required_columns = {
        "month",
        "tickets",
        "avg_resolution_hours",
        "staff_count",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    if len(df) < 2:
        raise ValueError(
            "CSV must contain at least two rows for trend analysis."
        )

    numeric_columns = [
        "tickets",
        "avg_resolution_hours",
        "staff_count",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must contain numeric values."
            )