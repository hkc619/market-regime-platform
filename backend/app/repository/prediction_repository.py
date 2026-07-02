from sqlalchemy import text
from sqlalchemy.orm import Session


def create_prediction_history(
    db: Session,
    *,
    ticker: str,
    as_of_date,
    predicted_class: int,
    predicted_regime: str,
    confidence: float,
    probabilities: dict,
    model_version: str,
    raw_window_rows: int,
    feature_rows: int,
    model_input_rows: int,
    feature_dim: int,
    input_start_date,
    input_end_date,
):
    query = text(
        """
        INSERT INTO prediction_history (
            ticker,
            as_of_date,
            predicted_class,
            predicted_regime,
            confidence,
            prob_trending_down,
            prob_transition_down,
            prob_transition_up,
            prob_trending_up,
            model_version,
            raw_window_rows,
            feature_rows,
            model_input_rows,
            feature_dim,
            input_start_date,
            input_end_date
        )
        VALUES (
            :ticker,
            :as_of_date,
            :predicted_class,
            :predicted_regime,
            :confidence,
            :prob_trending_down,
            :prob_transition_down,
            :prob_transition_up,
            :prob_trending_up,
            :model_version,
            :raw_window_rows,
            :feature_rows,
            :model_input_rows,
            :feature_dim,
            :input_start_date,
            :input_end_date
        )
        RETURNING
            id,
            ticker,
            as_of_date,
            predicted_class,
            predicted_regime,
            confidence,
            prob_trending_down,
            prob_transition_down,
            prob_transition_up,
            prob_trending_up,
            model_version,
            raw_window_rows,
            feature_rows,
            model_input_rows,
            feature_dim,
            input_start_date,
            input_end_date,
            created_at;
        """
    )

    result = db.execute(
        query,
        {
            "ticker": ticker.upper(),
            "as_of_date": as_of_date,
            "predicted_class": predicted_class,
            "predicted_regime": predicted_regime,
            "confidence": confidence,
            "prob_trending_down": probabilities["Trending-Down"],
            "prob_transition_down": probabilities["Transition-Down"],
            "prob_transition_up": probabilities["Transition-Up"],
            "prob_trending_up": probabilities["Trending-Up"],
            "model_version": model_version,
            "raw_window_rows": raw_window_rows,
            "feature_rows": feature_rows,
            "model_input_rows": model_input_rows,
            "feature_dim": feature_dim,
            "input_start_date": input_start_date,
            "input_end_date": input_end_date,
        },
    ).mappings().one()

    db.commit()

    return dict(result)

def get_latest_prediction_by_ticker(
    db: Session,
    ticker: str,
):
    query = text(
        """
        SELECT
            id,
            ticker,
            as_of_date,
            predicted_class,
            predicted_regime,
            confidence,
            prob_trending_down,
            prob_transition_down,
            prob_transition_up,
            prob_trending_up,
            model_version,
            raw_window_rows,
            feature_rows,
            model_input_rows,
            feature_dim,
            input_start_date,
            input_end_date,
            created_at
        FROM prediction_history
        WHERE ticker = :ticker
        ORDER BY created_at DESC
        LIMIT 1;
        """
    )

    result = db.execute(
        query,
        {"ticker": ticker.upper()},
    ).mappings().first()

    return dict(result) if result else None