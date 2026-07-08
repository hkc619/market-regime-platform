from sqlalchemy import text
from sqlalchemy.orm import Session

def get_latest_ohlcv(
    ticker: str, 
    db: Session
    ):

    ticker = ticker.upper().strip()
    
    query = text(
        """
        SELECT
            MAX(date) AS latest_date
        FROM market_prices
        WHERE ticker = :ticker;
        """
    )
    
    row = db.execute(query, {"ticker": ticker}).mappings().first()

    if row is None:
        return None

    return row['latest_date']


def upsert_market_prices(
    db: Session,
    rows: list[dict],
) -> int:
    if not rows:
        return 0

    query = text(
        """
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
        VALUES (
            :ticker,
            :date,
            :open,
            :high,
            :low,
            :close,
            :adjusted_close,
            :volume,
            :source
        )
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

    db.execute(query, rows)
    db.commit()

    return len(rows)
    

