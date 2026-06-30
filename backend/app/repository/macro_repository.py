from sqlalchemy import text
from sqlalchemy.orm import Session

def get_macro_daily(db: Session):
    query = text(
        """
        SELECT 
            date,
            vix,
            yield_10yr,
            yield_2yr
        FROM macro_daily
        """)
    rows = db.execute(query).mappings().all()

    return rows

def get_macro_monthly(db: Session):
    query = text(
    """
    SELECT 
        period,
        cpi_yoy_level
    FROM macro_monthly
    """)

    rows = db.execute(query).mappings().all()

    return rows
