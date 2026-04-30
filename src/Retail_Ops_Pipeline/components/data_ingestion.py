import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Retail_Ops_Pipeline.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

TRAIN_PATH = Path("data/processed/train.parquet")
TEST_PATH  = Path("data/processed/test.parquet")


class DataIngestion:

    def get_engine(self):
        pwd = quote_plus(os.getenv("MYSQL_PASSWORD"))
        url = (
            f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{pwd}"
            f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}"
            f"/{os.getenv('MYSQL_DATABASE')}"
        )
        return create_engine(url)

    def fetch_data(self) -> pd.DataFrame:
        if TRAIN_PATH.exists() and TEST_PATH.exists():
            logger.info("Parquet already exists — skipping DB fetch")
            return pd.read_parquet(TRAIN_PATH)

        query = """
            SELECT t.*, 
                   s.StoreType, s.Assortment, s.CompetitionDistance,
                   s.CompetitionOpenSinceMonth, s.CompetitionOpenSinceYear,
                   s.Promo2, s.Promo2SinceWeek, s.Promo2SinceYear, s.PromoInterval
            FROM raw_sales t
            LEFT JOIN store_info s ON t.Store = s.Store
            WHERE t.Open = 1 AND t.Sales > 0
            ORDER BY t.Date
        """
        logger.info("Fetching from MySQL...")
        df = pd.read_sql(query, self.get_engine())
        logger.info(f"Fetched {len(df):,} rows x {df.shape[1]} cols")
        return df

    def split_and_save(self, df: pd.DataFrame):
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        split    = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

        Path("data/processed").mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(TRAIN_PATH, index=False)
        test_df.to_parquet(TEST_PATH,  index=False)

        logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
        return train_df, test_df

    def run(self):
        df = self.fetch_data()
        if TRAIN_PATH.exists() and TEST_PATH.exists():
            return pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)
        return self.split_and_save(df)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train, test = ingestion.run()
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")