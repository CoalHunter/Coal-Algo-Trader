"""
=============================================================================
COAL ALGO-TRADER v1.0 — Dual-Engine Recommendation System
=============================================================================
Logic: Scan multi-source Newcastle Coal sampai berhasil. 
       Jika semua gagal, STOP dan lapor error (NO DUMMY DATA).
=============================================================================
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("CoalAlgoTrader")

COAL_TICKERS = ["ITMG.JK", "PTBA.JK", "ADRO.JK", "PTCO.JK", "BUMI.JK", "MBAP.JK"]
ICE_FLOOR_PRICE_USD = 120.0
OUTPUT_FILE = "coal_data.json"

# Bobot & Parameter (Sama seperti sebelumnya)
SWING_WEIGHTS = {"ice_momentum": 0.40, "technical": 0.40, "volume_sentiment": 0.20}
DIVIDEND_WEIGHTS = {"fundamental_safety": 0.40, "ice_trend_barrier": 0.40, "fear_panic_detector": 0.20}


# =============================================================================
# DATA FETCHING — MULTI-SOURCE SCANNER
# =============================================================================

class DataFetcher:
    """Scan satu per satu sumber data Newcastle Coal sampai berhasil."""

    @staticmethod
    def fetch_ice_newcastle(period: str = "6mo") -> Optional[pd.DataFrame]:
        logger.info("=" * 60)
        logger.info("MEMULAI SCAN DATA NEWCASTLE COAL...")
        logger.info("=" * 60)

        # 1. Coba yfinance
        df = DataFetcher._try_yfinance(period)
        if df is not None:
            return df

        # 2. Coba investpy
        df = DataFetcher._try_investpy(period)
        if df is not None:
            return df

        # 3. Jika semua gagal, return None (TIDAK ADA DUMMY DATA)
        logger.critical("SEMUA SUMBER DATA GAGAL. Tidak ada data Newcastle Coal yang valid.")
        return None

    @staticmethod
    def _try_yfinance(period: str) -> Optional[pd.DataFrame]:
        logger.info("\n[1/2] Mencoba sumber: yfinance...")
        tickers = ["^NCF=F", "NCF=F", "NC=F"]
        
        for ticker in tickers:
            try:
                logger.info(f"  -> Mengecek ticker: {ticker}")
                df = yf.Ticker(ticker).history(period=period)
                if not df.empty and len(df) > 10:
                    logger.info(f"  [OK] Berhasil mengambil data dari yfinance ({ticker})!")
                    return df
            except Exception as e:
                logger.warning(f"  [GAGAL] {ticker}: {e}")
        return None

    @staticmethod
    def _try_investpy(period: str) -> Optional[pd.DataFrame]:
        logger.info("\n[2/2] Mencoba sumber: investpy...")
        try:
            import investpy
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            logger.info(f"  -> Mengecek komoditas: newcastle coal")
            df = investpy.get_commodity_historical_data(
                commodity='newcastle coal',
                from_date=start_date.strftime('%d/%m/%Y'),
                to_date=end_date.strftime('%d/%m/%Y'),
                country='world'
            )
            
            if not df.empty and len(df) > 10:
                df = df.rename(columns={'Price': 'Close'})
                logger.info(f"  [OK] Berhasil mengambil data dari investpy!")
                return df
        except ImportError:
            logger.warning("  [GAGAL] Library investpy belum terinstall.")
        except Exception as e:
            logger.warning(f"  [GAGAL] investpy: {e}")
        return None

    @staticmethod
    def fetch_stock_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(ticker).history(period=period)
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    @staticmethod
    def fetch_stock_info(ticker: str) -> Optional[Dict]:
        try:
            info = yf.Ticker(ticker).info
            return info if info and len(info) > 5 else None
        except Exception as e:
            logger.error(f"Error fetching info {ticker}: {e}")
            return None


# =============================================================================
# ANALYSIS MODULES (Technical, Fundamental, Engines)
# =============================================================================
# (Singkatnya: Class TechnicalIndicators, ICEAnalysis, FundamentalAnalysis, 
# SwingEngine, dan DividendEngine tetap sama seperti versi sebelumnya. 
# Pastikan Anda menyalin class-class tersebut dari versi sebelumnya ke sini).

class TechnicalIndicators:
    # ... (Salin kode TechnicalIndicators dari versi sebelumnya) ...
    @staticmethod
    def calculate_sma(series, period): return series.rolling(window=period).mean()
    @staticmethod
    def calculate_rsi(series, period=14): 
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    # ... tambahkan method lain sesuai kebutuhan ...

class ICEAnalysis:
    # ... (Salin kode ICEAnalysis dari versi sebelumnya) ...
    @staticmethod
    def get_current_price(ice_df): return float(ice_df["Close"].iloc[-1]) if ice_df is not None and not ice_df.empty else None
    # ... tambahkan method lain ...

class FundamentalAnalysis:
    # ... (Salin kode FundamentalAnalysis dari versi sebelumnya) ...
    pass

class SwingEngine:
    # ... (Salin kode SwingEngine dari versi sebelumnya) ...
    pass

class DividendEngine:
    # ... (Salin kode DividendEngine dari versi sebelumnya) ...
    pass


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class CoalAlgoTrader:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.results = {
            "metadata": {"system": "Coal Algo-Trader v1.0", "generated_at": datetime.now().isoformat()},
            "ice_newcastle": {}, "swing_mode": [], "dividend_mode": [], "alerts": []
        }

    def run(self):
        logger.info("COAL ALGO-TRADER v1.0 — Starting...")

        # 1. SCAN DATA NEWCASTLE
        ice_df = self.fetcher.fetch_ice_newcastle()

        # 2. CEK KEBERHASILAN
        if ice_df is None:
            error_msg = "GAGAL MENGAMBIL DATA NEWCASTLE COAL. Semua sumber (yfinance, investpy) tidak berhasil mengembalikan data."
            logger.critical(error_msg)
            
            # Tulis status gagal ke JSON
            self.results["ice_newcastle"] = {
                "status": "FAILED",
                "error_message": error_msg,
                "current_price_usd": None
            }
            self.results["alerts"].append({
                "type": "CRITICAL_DATA_FAILURE",
                "message": error_msg,
                "severity": "CRITICAL"
            })
            
            # Force semua sinyal ke NO_DATA agar user tidak salah ambil tindakan
            for ticker in COAL_TICKERS:
                self.results["swing_mode"].append({"ticker": ticker, "signal": "NO_DATA", "reason": "ICE Data Failed"})
                self.results["dividend_mode"].append({"ticker": ticker, "signal": "NO_DATA", "reason": "ICE Data Failed"})
            
            self._export()
            logger.critical("PROSES DIHENTIKAN. Tidak ada analisis yang dilakukan karena data komoditas gagal diambil.")
            return # STOP DI SINI

        # 3. JIKA BERHASIL, LANJUTKAN ANALISIS SEPERTI BIASA
        logger.info("Data Newcastle berhasil diambil. Melanjutkan analisis saham...")
        ice_current = ICEAnalysis.get_current_price(ice_df)
        
        self.results["ice_newcastle"] = {
            "status": "SUCCESS",
            "current_price_usd": round(ice_current, 2),
            "floor_threshold_usd": ICE_FLOOR_PRICE_USD,
            "above_floor": ice_current >= ICE_FLOOR_PRICE_USD,
            "momentum": ICEAnalysis.calculate_momentum_score(ice_df),
            "trend_barrier": ICEAnalysis.calculate_trend_barrier_score(ice_df),
        }

        if ice_current < ICE_FLOOR_PRICE_USD:
            self.results["alerts"].append({
                "type": "ICE_FLOOR_BREACH",
                "message": f"ICE Newcastle (${ice_current:.2f}) di bawah floor (${ICE_FLOOR_PRICE_USD})",
                "severity": "HIGH"
            })

        swing_engine = SwingEngine(ice_df)
        dividend_engine = DividendEngine(ice_df)

        for ticker in COAL_TICKERS:
            stock_df = self.fetcher.fetch_stock_history(ticker)
            info = self.fetcher.fetch_stock_info(ticker)

            self.results["swing_mode"].append(swing_engine.analyze_stock(ticker, stock_df))
            self.results["dividend_mode"].append(dividend_engine.analyze_stock(ticker, stock_df, info))

        self._export()

    def _export(self):
        with open(OUTPUT_FILE, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Data berhasil diekspor ke {OUTPUT_FILE}")


if __name__ == "__main__":
    CoalAlgoTrader().run()
