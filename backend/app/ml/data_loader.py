import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
data_path = os.getenv("DATA_PATH")
print("\n" + "=" * 70)
print("  SECTION 1: LOADING DATA")
print("=" * 70)

def load_ohlcv(sheet, close_col="Close", date_col="Date"):
    df = pd.read_excel(data_path, sheet_name=sheet, parse_dates=[date_col])
    df = df.rename(columns={date_col: "Date"})
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df

## spy_close, spy_vol, spy_high, spy_low, qqq_close, tlt_close
def dropna_ohlcv(sheet, col):
    df = load_ohlcv(sheet)
    return df[col].dropna()

def load_macro_daily():
    macro_daily_df = pd.read_excel(data_path, sheet_name="Macro_Daily", header=0)
    vix_df  = macro_daily_df.iloc[:, [0, 1]].copy()
    yr10_df = macro_daily_df.iloc[:, [3, 4]].copy()
    yr2_df  = macro_daily_df.iloc[:, [6, 7]].copy()
    def clean_macro_pair(df):
        df.columns = ["Date", "Value"]
        df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        return df["Value"].dropna()
    
    vix_s  = clean_macro_pair(vix_df).rename("VIX")
    yr10_s = clean_macro_pair(yr10_df).rename("Yield10yr")
    yr2_s  = clean_macro_pair(yr2_df).rename("Yield2yr")
    
    return vix_s, yr10_s, yr2_s

def load_cpi():

    cpi_df = pd.read_excel(data_path, sheet_name="Macro_Monthly")
    cpi_df.columns = ["Date", "CPI_YoY"]
    cpi_df["Date"] = pd.to_datetime(cpi_df["Date"], errors="coerce")
    cpi_s  = cpi_df.dropna(subset=["Date"]).set_index("Date").sort_index()["CPI_YoY"]
    return cpi_s

print(f"  QQQ / TLT  loaded")
print(f"  Macro: VIX, 10yr, 2yr, CPI loaded")