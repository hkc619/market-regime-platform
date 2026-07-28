import time
from decimal import Decimal
from typing import Any
from datetime import date

from app.db.session import get_db
from app.core.logging import get_logger
from app.core.model_config import RAW_LOOKBACK_DAYS, MODEL_VERSION
from app.core.exceptions import (
    InsufficientFeatureDataError,
    InsufficientRawDataError,
    ModelInferenceError,
    TickerNotFoundError,
)

from app.services.data_service import (
    get_latest_support_window, 
    get_macro_daily_window, 
    get_macro_monthly_window, 
    )

from app.services.backtest_data_service import get_range_ticker_prices

logger = get_logger(__name__)

def predict_for_date(
        ticker,
        model_state,
        request_id,
        ticker_rows,
        sup0_rows,
        sup1_rows,
        macro_daily_rows,
        macro_monthly_rows
) :
    """
    Run model inference using data available up to as_of_date.

    persist=False is used by backtesting so historical predictions
    are not written into the production prediction history table.
    """
    
    from app.services.inference_db_service import prepare_backtest_inference_input
    from app.services.inference import predict_proba
    from app.services.features_input_service import build_latest_model_input

    start_time = time.perf_counter()
    ticker = ticker.upper().strip()

    logger.info(
        "Prediction request started | request_id=%s | ticker=%s",
        request_id,
        ticker,   
    )

    model = model_state.model
    metadata = model_state.metadata
    device = model_state.device
    try: 
        raw = prepare_backtest_inference_input(
            ticker=ticker,
            ticker_rows=ticker_rows, 
            sup0_rows=sup0_rows, 
            sup1_rows=sup1_rows,
            macro_daily_rows=macro_daily_rows,
            macro_monthly_rows=macro_monthly_rows
        )

        raw_rows = len(raw.ticker_close)
        
        logger.info("Raw inference loaded | request_id=%s | ticker=%s | raw_rows=%d ",
            request_id,
            ticker,
            raw_rows,
        )
        if raw_rows == 0:
            raise TickerNotFoundError(f"No market data found for ticker={ticker}")
        
        if raw_rows < RAW_LOOKBACK_DAYS:
            raise InsufficientRawDataError(
                f"{ticker} only has {raw_rows} raw rows. "
                f"Need at least {RAW_LOOKBACK_DAYS}"
            )
        
        model_input = build_latest_model_input(raw)

        logger.info(
            "Model input built | request_id=%s | ticker=%s | feature_rows=%s | input_shape=%s",
            request_id,
            ticker,
            model_input.feature_rows,
            model_input.latest_60_feat.shape,
        )

        if model_input.latest_60_feat.shape[0] < 60:
            raise InsufficientFeatureDataError(
                f"{ticker} does not have enough valid feature rows for inference."
        )

        try:
            prediction = predict_proba(model_input, device, model, metadata["model_config"]["scaler_path"])
        except Exception as e:
            logger.exception(
                "Model inference failed | request_id=%s | ticker=%s",
                request_id,
                ticker,
            )
            raise ModelInferenceError(str(e)) from e

        return {
            "ticker":ticker,
            "as_of_date":model_input.end_date,
            "predicted_class":int(prediction["predicted_class"]),
            "predicted_regime":prediction["predicted_regime"],
            "confidence":float(prediction["confidence"]),
            "raw_window_rows": len(raw.ticker_close),
            "model_input_rows":model_input.latest_60_feat.shape[0],
            "feature_dim":model_input.feature_dim,
            "input_start_date":model_input.start_date,
            "input_end_date":model_input.end_date,
        }
    except:
        pass
        
        
        



def backtest_for_range(
        model_state,
        db,
        ticker,
        sup0,
        sup1,
        start_date: date,
        end_date: date,
        request_id
):
    ticker_rows = get_range_ticker_prices(
        db=db,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback=312
    )
    sup0_rows = get_latest_support_window(
        db=db,
        support=sup0,
        start_date=start_date,
        end_date=end_date,

    )
    sup1_rows = get_latest_support_window(
        db=db,
        support=sup1,
        start_date=start_date,
        end_date=end_date,
    )
    macro_daily_rows = get_macro_daily_window(
        db=db
    )
    macro_monthly_rows = get_macro_monthly_window(
        db=db
    ) 

    delta = (end_date - start_date).days + 1
    row = []
    for i in range(delta):
        start = i
        end = 312 + i + 1
        row.append(predict_for_date(
            ticker,
            model_state=model_state,
            request_id=request_id,
            ticker_rows=ticker_rows[start:end],
            sup0_rows=sup0_rows[start:end],
            sup1_rows=sup1_rows[start:end],
            macro_daily_rows=macro_daily_rows,
            macro_monthly_rows=macro_monthly_rows
        ))
    return row


    logger.info(
               "Backtest completed | request_id=%s | ticker=%s",
                request_id,
                ticker,
            )



# if __name__ == "__main__":
#     db = next(get_db())
#     try:
#         rows = backtest_for_range(
#             db=db,
#             ticker="SPY",
#             sup0="QQQ",
#             sup1="TLT",
#             start_date=date(2025, 12, 1),
#             end_date=date(2025, 12, 3),
#             request_id="Manual Test"
#         )
#     finally:
#         db.close()
