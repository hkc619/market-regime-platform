from sqlalchemy import text
from sqlalchemy.orm import Session
import pandas as pd

def get_latest_macro_daily(db: Session):

    query =  text("""
                SELECT MAX(date)
                FROM macro_daily;
            """)
    row = db.execute(query).scalar()

    if row is None:
        return None

    return row

def get_latest_macro_monthly(db: Session):
    
    query =  text("""
                SELECT MAX(period)
                FROM macro_monthly;
            """)
    row = db.execute(query).scalar()

    if row is None:
        return None

    return row


def upsert_macro_daily(
        db: Session,
        macro_df: pd.DataFrame,
) -> int:
    
    macro_df = macro_df.astype(object).where(pd.notna(macro_df), None)
    records = macro_df.to_dict(orient="records")

    if not records:
        return 0

    query = text("""
            INSERT INTO macro_daily (
                date,
                vix,
                yield_10yr,
                yield_2yr,
                source
            )
            VALUES (
                :date,
                :vix,
                :yield_10yr,
                :yield_2yr,
                :source
            )
            ON CONFLICT (date)
            DO UPDATE SET
                vix = EXCLUDED.vix,
                yield_10yr = EXCLUDED.yield_10yr,
                yield_2yr = EXCLUDED.yield_2yr,
                source = EXCLUDED.source,
                updated_at = NOW()
        """)

    db.execute(query, records)
    db.commit()

    return len(records)


def upsert_macro_monthly(
        db: Session,
        macro_df: pd.DataFrame,
) -> int:
    records = macro_df.to_dict(orient="records")

    if not records:
        return 0

    query = text("""
            INSERT INTO macro_monthly (
                period,
                cpi_yoy_level,
                source
            )
            VALUES (
                :date,
                :cpi_yoy_level,
                :source
            )
            ON CONFLICT (period)
            DO UPDATE SET
                cpi_yoy_level = EXCLUDED.cpi_yoy_level,
                source = EXCLUDED.source,
                updated_at = NOW()
        """)

    db.execute(query, records)
    db.commit()

    return len(records)