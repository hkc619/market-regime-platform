from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


RAW_DATA_DIR = Path("/Users/hkc619/Documents/PY/project/market-regime-platform/data/raw/ohlcv")
STAGING_TABLE = "_staging_market_prices"


REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns like:
    Ticker, Date, Open, High, Low, Close, Volume

    into:
    ticker, date, open, high, low, close, volume
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def read_one_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # if there is no ticker column, refer the ticker from file name
    if "ticker" not in df.columns:
        df["ticker"] = csv_path.stem.upper()

    # check if there is missing column 
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    # keep essential columns
    df = df[list(REQUIRED_COLUMNS)].copy()

    # ticker formatting
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # datatime 
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # numeric value
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    # set adjusted price as close price
    df["adjusted_close"] = df["close"]

    # data source
    df["source"] = csv_path.name

    return df


def validate_market_prices(df: pd.DataFrame) -> None:
    """
    Basic data quality checks before inserting into PostgreSQL.
    """
    if df.empty:
        raise ValueError("No market price data found.")

    required_for_insert = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    null_counts = df[required_for_insert].isna().sum()
    bad_nulls = null_counts[null_counts > 0]

    if not bad_nulls.empty:
        raise ValueError(f"Found missing values:\n{bad_nulls}")

    duplicated = df.duplicated(subset=["ticker", "date"], keep=False)
    if duplicated.any():
        duplicates = df.loc[duplicated, ["ticker", "date"]].sort_values(
            ["ticker", "date"]
        )
        raise ValueError(f"Found duplicated ticker/date rows:\n{duplicates}")

    invalid_price = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
    )

    if invalid_price.any():
        bad_rows = df.loc[invalid_price, ["ticker", "date", "open", "high", "low", "close"]]
        raise ValueError(f"Found non-positive prices:\n{bad_rows}")

    invalid_ohlc = df["high"] < df["low"]
    if invalid_ohlc.any():
        bad_rows = df.loc[invalid_ohlc, ["ticker", "date", "high", "low"]]
        raise ValueError(f"Found rows where high < low:\n{bad_rows}")


def load_all_market_price_csvs() -> pd.DataFrame:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    frames = []

    for csv_path in csv_files:
        print(f"Reading {csv_path}")
        frames.append(read_one_csv(csv_path))

    df = pd.concat(frames, ignore_index=True)

    # sorting for easily debug
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    validate_market_prices(df)

    return df


def upsert_market_prices(df: pd.DataFrame) -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set in .env")

    engine = create_engine(database_url)

    insert_columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
    ]

    df_to_insert = df[insert_columns].copy()

    with engine.begin() as conn:
        # 1. 建立 staging table
        df_to_insert.to_sql(
            STAGING_TABLE,
            con=conn,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )

        # 2. 從 staging table upsert 到正式 table
        conn.execute(
            text(
                f"""
                INSERT INTO market_prices (
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    adjusted_close,
                    volume,
                    source
                )
                SELECT
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    adjusted_close,
                    volume,
                    source
                FROM {STAGING_TABLE}
                ON CONFLICT (ticker, date)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adjusted_close = EXCLUDED.adjusted_close,
                    volume = EXCLUDED.volume,
                    source = EXCLUDED.source,
                    updated_at = NOW();
                """
            )
        )

        # 3. 刪掉 staging table
        conn.execute(text(f"DROP TABLE IF EXISTS {STAGING_TABLE};"))

    print(f"Upserted {len(df_to_insert)} rows into market_prices.")


def main() -> None:
    df = load_all_market_price_csvs()

    print("Preview:")
    print(df.head())

    print("\nRow count by ticker:")
    print(df.groupby("ticker").size())

    upsert_market_prices(df)


if __name__ == "__main__":
    main()