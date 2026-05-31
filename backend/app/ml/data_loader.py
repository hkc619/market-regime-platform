import pandas as pd
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

## spy_close, spy_vol, spy_high, spy_low, qqq_close, tlt_close
## combine with load_ohlcv and dropna_ohlcv
def load_ohlcv(path, sheet, drop_col, close_col="Close", date_col="Date"):
    df = pd.read_excel(path, sheet_name=sheet, parse_dates=[date_col])
    df = df.rename(columns={date_col: "Date"})
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df[drop_col].dropna()

def load_macro_daily(path):
    macro_daily_df = pd.read_excel(
        path, 
        sheet_name="Macro_Daily", 
        header=0
    )

    def clean_macro_pair(date_col, value_col, name):
        df = macro_daily_df.iloc[:, [date_col, value_col]].copy()
        df.columns = ["Date", "Value"]

        df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
        df = (
            df
              .dropna(subset=["Date", "Value"])
              .set_index("Date")
              .sort_index()
        )
        return df["Value"].rename(name)
    
    vix_s  = clean_macro_pair(0, 1, "VIX")
    yr10_s = clean_macro_pair(3, 4, "Yield10yr")
    yr2_s  = clean_macro_pair(6, 7, "Yield2yr")
    
    return vix_s, yr10_s, yr2_s

def load_cpi(path):
    cpi_df = pd.read_excel(path, sheet_name="Macro_Monthly")
    cpi_df.columns = ["Date", "CPI_YoY"]
    cpi_df["Date"] = pd.to_datetime(cpi_df["Date"], errors="coerce")
    cpi_s  = cpi_df.dropna(subset=["Date"]).set_index("Date").sort_index()["CPI_YoY"]
    return cpi_s
