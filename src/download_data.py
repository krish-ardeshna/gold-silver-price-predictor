import logging
import yfinance as yf
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def download_futures_data(ticker, name, period="10y", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data")
    try:
        logger.info(f"Downloading {name.upper()} data")
        data = yf.download(ticker, period=period, progress=False)

        if data.empty:
            logger.error(f"No data for {ticker}")
            return False

        # Harden against unforeseen yfinance API updates (MultiIndex flattening)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data.columns = data.columns.droplevel(1)
            else:
                data.columns = data.columns.droplevel(0)

        logger.info(f"Downloaded {len(data)} rows")
        data.reset_index(inplace=True)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}.csv")
        data.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False

def main():
    logger.info("Starting download")
    success = 0

    if download_futures_data("GC=F", "gold"):
        success += 1
    if download_futures_data("SI=F", "silver"):
        success += 1

    logger.info(f"Complete: {success}/2 successful")

    if success == 2:
        logger.info("All data downloaded")
        return 0
    else:
        logger.warning("Some failed")
        return 1

if __name__ == "__main__":
    exit(main())