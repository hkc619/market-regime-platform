from sqlalchemy import text
from sqlalchemy.orm import Session


def create_data_update_log(
    db: Session,
    *,
    request_id: str | None,
    data_type: str,
    source: str,
    ticker: str | None,
    start_date,
    end_date,
    status: str,
    rows_fetched: int = 0,
    rows_inserted_or_updated: int = 0,
    error_message: str | None = None,
) -> dict:
    query = text(
        """
        INSERT INTO data_update_log (
            request_id,
            data_type,
            source,
            ticker,
            start_date,
            end_date,
            status,
            rows_fetched,
            rows_inserted_or_updated,
            error_message
        )
        VALUES (
            :request_id,
            :data_type,
            :source,
            :ticker,
            :start_date,
            :end_date,
            :status,
            :rows_fetched,
            :rows_inserted_or_updated,
            :error_message
        )
        RETURNING
            id,
            request_id,
            data_type,
            source,
            ticker,
            start_date,
            end_date,
            status,
            rows_fetched,
            rows_inserted_or_updated,
            error_message,
            created_at;
        """
    )

    result = db.execute(
        query,
        {
            "request_id": request_id,
            "data_type": data_type,
            "source": source,
            "ticker": ticker.upper() if ticker else None,
            "start_date": start_date,
            "end_date": end_date,
            "status": status,
            "rows_fetched": rows_fetched,
            "rows_inserted_or_updated": rows_inserted_or_updated,
            "error_message": error_message,
        },
    ).mappings().one()

    db.commit()

    return dict(result)

def get_recent_data_update_logs(
    db: Session,
    *,
    ticker: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    conditions = []
    params = {"limit": limit}

    if ticker:
        conditions.append("ticker = :ticker")
        params["ticker"] = ticker.upper()

    if status:
        conditions.append("status = :status")
        params["status"] = status

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    query = text(
        f"""
        SELECT
            id,
            request_id,
            data_type,
            source,
            ticker,
            start_date,
            end_date,
            status,
            rows_fetched,
            rows_inserted_or_updated,
            error_message,
            created_at
        FROM data_update_log
        {where_clause}
        ORDER BY created_at DESC
        LIMIT :limit;
        """
    )

    rows = db.execute(query, params).mappings().all()

    return [dict(row) for row in rows]