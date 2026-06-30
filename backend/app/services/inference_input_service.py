import pandas as pd

def market_rows_to_dataframe(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


