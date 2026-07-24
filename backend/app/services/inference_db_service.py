from app.services.data_service import get_latest_ticker_window, get_latest_support_window, get_macro_daily_window, get_macro_monthly_window
from app.services.inference_input_service import market_rows_to_dataframe, macro_rows_to_dataframe
from app.core.config import RawInferenceSeries

def prepare_inference_input(
        db, 
        ticker: str,
        latest: bool, 
        sup0: str, 
        sup1: str, 
        lookback: int = 312
    ):
    if latest:
        ticker_rows = get_latest_ticker_window(
        db=db,
        ticker=ticker,
        lookback=lookback,
        )
    else:
        ticker_rows = get_latest_ticker_window(
        db=db,
        ticker=ticker,
        lookback=lookback,
        )

    # 1. primary ticker OHLCV
    ticker_df = market_rows_to_dataframe(ticker_rows)
    ticker_close = ticker_df["close"]
    ticker_vol = ticker_df["volume"]
    ticker_high = ticker_df["high"]
    ticker_low = ticker_df["low"]
    
    # 2. supporting assets
    sup0_rows = get_latest_support_window(
    db=db,
    support=sup0,
    start_date=ticker_df.index.min(),
    end_date=ticker_df.index.max(),
    )

    sup1_rows = get_latest_support_window(
    db=db,
    support=sup1,
    start_date=ticker_df.index.min(),
    end_date=ticker_df.index.max(),
    )

    sup0_df = market_rows_to_dataframe(sup0_rows)
    sup1_df = market_rows_to_dataframe(sup1_rows)

    sup0_close = sup0_df["close"].reindex(ticker_df.index).ffill()
    sup1_close = sup1_df["close"].reindex(ticker_df.index).ffill()

    # 3. macro daily
    macro_daily_rows = get_macro_daily_window(db)
    macro_daily_df = macro_rows_to_dataframe(macro_daily_rows)

    vix_s = macro_daily_df["vix"].reindex(ticker_df.index).ffill()
    yr10_s = macro_daily_df["yield_10yr"].reindex(ticker_df.index).ffill()
    yr2_s = macro_daily_df["yield_2yr"].reindex(ticker_df.index).ffill()

    # 4. macro monthly
    macro_monthly_rows = get_macro_monthly_window(db)
    macro_monthly_df = macro_rows_to_dataframe(macro_monthly_rows)
    cpi_s = macro_monthly_df["cpi_yoy_level"].reindex(ticker_df.index).ffill()

    
    return RawInferenceSeries(
        ticker=ticker,
        ticker_close=ticker_close,
        ticker_vol=ticker_vol,
        ticker_high=ticker_high,
        ticker_low=ticker_low,
        sup0_close=sup0_close,
        sup1_close=sup1_close,
        vix_s=vix_s,
        yr10_s=yr10_s,
        yr2_s=yr2_s,
        cpi_s=cpi_s,
        idx=ticker_df.index,
        )

    