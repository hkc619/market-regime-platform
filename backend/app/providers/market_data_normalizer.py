import pandas as pd


def normalize_market_data(
    df: pd.DataFrame,
    ticker: str,
    source: str,
) -> list[dict]:
    if df.empty:
        return []

    df = df.reset_index()

    df.columns = [str(col[0]).strip().lower().replace(" ", "_") for col in df.columns]


    # datatime or date
    date_col = "date" if "date" in df.columns else "datetime"

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "ticker": ticker.upper(),
                "date": pd.to_datetime(row[date_col]).date(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "adjusted_close": float(row.get("adj_close", row["close"])),
                "volume": int(row["volume"]),
                "source": source,
            }
        )
    return rows