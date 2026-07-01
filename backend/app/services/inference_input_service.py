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
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df

def macro_rows_to_dataframe(rows)-> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"])
        df = df.sort_values("period").set_index("period")

    numeric_cols = [
        "vix",
        "yield_10yr",
        "yield_2yr",
        "cpi_yoy_level",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df

