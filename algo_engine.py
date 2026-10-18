"""
=============================================================================
COAL ALGO-TRADER v1.0 — Dual-Engine Recommendation System
=============================================================================
FINAL VERSION: World Bank API (FREE, NO API KEY) + Hardcoded Fallback
=============================================================================
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
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

# Swing Trading Parameters
SWING_MA_FAST = 10
SWING_MA_SLOW = 20
SWING_RSI_PERIOD = 14
SWING_RSI_OVERSOLD = 35
SWING_RSI_OVERBOUGHT = 70
SWING_TP_MIN_PCT = 5.0
SWING_TP_MAX_PCT = 10.0
SWING_SL_MIN_PCT = 3.0
SWING_SL_MAX_PCT = 5.0

# Dividend Investing Parameters
DIV_MA_TREND = 50
DIV_MA_LONG = 200
DIV_TARGET_FORWARD_YIELD_PCT = 12.0

SWING_WEIGHTS = {"ice_momentum": 0.40, "technical": 0.40, "volume_sentiment": 0.20}
DIVIDEND_WEIGHTS = {"fundamental_safety": 0.40, "ice_trend_barrier": 0.40, "fear_panic_detector": 0.20}


# =============================================================================
# DATA FETCHING — WORLD BANK API (PASTI WORK)
# =============================================================================

class DataFetcher:
    """Mengambil data Newcastle Coal. World Bank API = GRATIS, NO API KEY."""

    @staticmethod
    def fetch_ice_newcastle(period: str = "6mo") -> Optional[pd.DataFrame]:
        logger.info("=" * 60)
        logger.info("MEMULAI SCAN DATA NEWCASTLE COAL...")
        logger.info("=" * 60)

        # Source 1: yfinance
        df = DataFetcher._try_yfinance(period)
        if df is not None:
            return df

        # Source 2: World Bank API (FREE, NO API KEY)
        df = DataFetcher._try_world_bank()
        if df is not None:
            return df

        # Source 3: Hardcoded fallback (terupdate manual)
        logger.warning("⚠️  Menggunakan data hardcoded sebagai fallback terakhir")
        return DataFetcher._get_hardcoded_data()

    @staticmethod
    def _try_yfinance(period: str) -> Optional[pd.DataFrame]:
        logger.info("\n[Source 1/3] yfinance")
        tickers = ["^NCF=F", "NCF=F", "NC=F"]
        for ticker in tickers:
            try:
                logger.info(f"  -> {ticker}")
                df = yf.Ticker(ticker).history(period=period)
                if not df.empty and len(df) > 10:
                    logger.info(f"  ✅ SUCCESS: {ticker} ({len(df)} rows)")
                    return df
            except Exception as e:
                logger.warning(f"  ❌ {ticker}: {e}")
        logger.info("  yfinance: Gagal")
        return None

    @staticmethod
    def _try_world_bank() -> Optional[pd.DataFrame]:
        """
        World Bank Commodity Price API - GRATIS, NO API KEY.
        Data: Coal, Australian (USD/ton) - kode: PCOALMTUSDM
        """
        logger.info("\n[Source 2/3] World Bank API (FREE)")
        
        try:
            # World Bank API endpoint untuk harga batu bara
            url = "https://api.worldbank.org/v2/commodity/price/monthly/PCOALMTUSDM"
            params = {
                "format": "json",
                "per_page": 200,
                "date": "2020M01:2026M12"
            }
            
            logger.info("  -> Fetching dari World Bank API...")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # World Bank returns [metadata, [data_array]]
                if isinstance(data, list) and len(data) >= 2:
                    records = data[1]
                    
                    if records and len(records) > 0:
                        logger.info(f"  ✅ SUCCESS: World Bank API ({len(records)} records)")
                        
                        # Convert ke DataFrame
                        rows = []
                        for record in records:
                            date_str = record.get("date", "")  # format: "2024M01"
                            value = record.get("value")
                            if date_str and value is not None:
                                # Parse "2024M01" -> 2024-01-01
                                year = int(date_str[:4])
                                month = int(date_str[5:7])
                                date = datetime(year, month, 1)
                                rows.append({"date": date, "Close": float(value)})
                        
                        df = pd.DataFrame(rows)
                        df = df.set_index("date")
                        df = df.sort_index()
                        
                        # Convert monthly ke daily (forward fill)
                        if len(df) > 0:
                            last_date = df.index[-1]
                            daily_index = pd.date_range(start=df.index[0], end=last_date, freq="D")
                            df_daily = df.reindex(daily_index, method="ffill")
                            df_daily["Volume"] = 1000
                            logger.info(f"  ✅ Converted to {len(df_daily)} daily rows")
                            return df_daily
                    else:
                        logger.warning("  ❌ World Bank: No records")
                else:
                    logger.warning(f"  ❌ World Bank: Unexpected response format")
            else:
                logger.warning(f"  ❌ World Bank: HTTP {response.status_code}")
                
        except Exception as e:
            logger.warning(f"  ❌ World Bank API error: {e}")
        
        return None

    @staticmethod
    def _get_hardcoded_data() -> pd.DataFrame:
        """
        Fallback terakhir: data hardcoded.
        UPDATE harga ini secara berkala dari:
        https://www.worldbank.org/en/research/commodity-markets
        """
        logger.info("\n[Source 3/3] Hardcoded fallback")
        
        # UPDATE HARGA INI SETIAP BULAN!
        # Cek di: https://pinksheet.worldbank.org/pinksheet/
        CURRENT_COAL_PRICE = 135.50  # USD/ton - UPDATE INI!
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)
        prices = [CURRENT_COAL_PRICE + np.random.normal(0, 1.5) for _ in range(100)]
        
        df = pd.DataFrame({
            "Close": prices,
            "Volume": [1000] * 100
        }, index=dates)
        
        logger.warning(f"  ⚠️  HARDCODED PRICE: ${CURRENT_COAL_PRICE}/ton")
        logger.warning(f"  ⚠️  UPDATE DI algo_engine.py line ~130!")
        return df

    @staticmethod
    def fetch_stock_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(ticker).history(period=period)
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Error {ticker}: {e}")
            return None

    @staticmethod
    def fetch_stock_info(ticker: str) -> Optional[Dict]:
        try:
            info = yf.Ticker(ticker).info
            return info if info and len(info) > 5 else None
        except Exception as e:
            logger.error(f"Error info {ticker}: {e}")
            return None


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        avg_gain = avg_gain.iloc[period:]
        avg_loss = avg_loss.iloc[period:]
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_rate_of_change(series: pd.Series, period: int = 14) -> pd.Series:
        return ((series - series.shift(period)) / series.shift(period)) * 100

    @staticmethod
    def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        avg_vol = volume.rolling(window=period, min_periods=period).mean()
        return volume / avg_vol.replace(0, np.nan)

    @staticmethod
    def detect_accumulation(df: pd.DataFrame, lookback: int = 10) -> Dict:
        if df is None or len(df) < lookback + 5:
            return {"accumulation_score": 0, "signal": "NO_DATA"}

        recent = df.tail(lookback)
        vol_ratio = recent["Volume"] / recent["Volume"].rolling(20).mean()
        vol_ratio = vol_ratio.fillna(1.0)

        obv = [0]
        for i in range(1, len(recent)):
            if recent["Close"].iloc[i] > recent["Close"].iloc[i - 1]:
                obv.append(obv[-1] + recent["Volume"].iloc[i])
            elif recent["Close"].iloc[i] < recent["Close"].iloc[i - 1]:
                obv.append(obv[-1] - recent["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        obv_series = pd.Series(obv, index=recent.index)

        score = 0
        signals = []
        price_change_pct = ((recent["Close"].iloc[-1] - recent["Close"].iloc[0]) / recent["Close"].iloc[0]) * 100
        high_vol_days = (vol_ratio > 1.5).sum()

        if high_vol_days >= 3 and abs(price_change_pct) < 5:
            score += 40
            signals.append("VOLUME_SPIKE_STABLE_PRICE")
        if len(obv_series) >= 5 and obv_series.iloc[-1] - obv_series.iloc[-5] > 0:
            score += 30
            signals.append("OBV_UPTREND")
        high_vol_mask = vol_ratio > 1.3
        if high_vol_mask.any() and (recent.loc[high_vol_mask, "Close"] > recent.loc[high_vol_mask, "Open"]).mean() > 0.6:
            score += 30
            signals.append("BUYING_PRESSURE_HIGH_VOL")

        if score >= 70: signal = "STRONG_ACCUMULATION"
        elif score >= 40: signal = "MODERATE_ACCUMULATION"
        else: signal = "NO_SIGNIFICANT_ACCUMULATION"

        return {
            "accumulation_score": score, "signal": signal,
            "signals_detail": signals, "high_vol_days": int(high_vol_days),
            "price_change_pct": round(price_change_pct, 2),
        }


# =============================================================================
# ICE ANALYSIS
# =============================================================================

class ICEAnalysis:
    @staticmethod
    def get_current_price(ice_df: pd.DataFrame) -> Optional[float]:
        if ice_df is None or ice_df.empty:
            return None
        return float(ice_df["Close"].iloc[-1])

    @staticmethod
    def calculate_momentum_score(ice_df: pd.DataFrame) -> Dict:
        if ice_df is None or len(ice_df) < 25:
            return {"score": 50, "roc_5d": 0, "roc_10d": 0, "roc_20d": 0, "trend": "UNKNOWN"}

        close = ice_df["Close"]
        roc_5 = TechnicalIndicators.calculate_rate_of_change(close, 5).iloc[-1]
        roc_10 = TechnicalIndicators.calculate_rate_of_change(close, 10).iloc[-1]
        roc_20 = TechnicalIndicators.calculate_rate_of_change(close, 20).iloc[-1]

        def roc_to_score(v):
            return (max(-15, min(15, v)) + 15) / 30 * 100

        composite = roc_to_score(roc_5) * 0.50 + roc_to_score(roc_10) * 0.30 + roc_to_score(roc_20) * 0.20
        ma_20 = close.rolling(20).mean().iloc[-1]
        current = close.iloc[-1]
        trend = "BULLISH" if current > ma_20 * 1.02 else "BEARISH" if current < ma_20 * 0.98 else "SIDEWAYS"

        return {
            "score": round(composite, 2), "roc_5d": round(float(roc_5), 2),
            "roc_10d": round(float(roc_10), 2), "roc_20d": round(float(roc_20), 2),
            "trend": trend, "current_price_usd": round(float(current), 2),
        }

    @staticmethod
    def calculate_trend_barrier_score(ice_df: pd.DataFrame) -> Dict:
        if ice_df is None or len(ice_df) < 50:
            return {"score": 50, "above_ma50": None, "above_ma200": None, "position": "NO_DATA", "current_price_usd": None}

        close = ice_df["Close"]
        ma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        above_ma50 = current > ma_50

        score = 70 if above_ma50 else 30
        if len(close) >= 200:
            ma_200 = close.rolling(200).mean().iloc[-1]
            above_ma200 = current > ma_200
            score = 0
            if above_ma200: score += 50
            if above_ma50: score += 30
            if ma_50 > ma_200: score += 20
            elif ma_50 < ma_200: score -= 10
            score = max(0, min(100, score))
            position = "STRONG_BULL" if (above_ma50 and above_ma200) else "WEAK_BULL" if above_ma200 else "RECOVERY" if above_ma50 else "BEARISH"
            return {"score": round(score, 2), "above_ma50": above_ma50, "above_ma200": above_ma200, "position": position, "ma50": round(float(ma_50), 2), "ma200": round(float(ma_200), 2), "current_price_usd": round(float(current), 2)}

        return {"score": score, "above_ma50": above_ma50, "above_ma200": None, "position": "ABOVE_MA50" if above_ma50 else "BELOW_MA50", "current_price_usd": round(float(current), 2)}

    @staticmethod
    def calculate_fear_panic_score(ice_df: pd.DataFrame, stock_df: pd.DataFrame) -> Dict:
        if stock_df is None or len(stock_df) < 50:
            return {"score": 0, "panic_detected": False, "opportunity": "NO_DATA"}

        close = stock_df["Close"]
        volume = stock_df["Volume"]
        ma_50_stock = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]
        discount = ((ma_50_stock - current_price) / ma_50_stock) * 100
        vol_spike = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if volume.rolling(20).mean().iloc[-1] > 0 else 1
        ice_roc_5 = 0
        if ice_df is not None and len(ice_df) >= 5:
            ice_roc_5 = ((ice_df["Close"].iloc[-1] - ice_df["Close"].iloc[-5]) / ice_df["Close"].iloc[-5]) * 100

        score = 0
        panic = False
        if discount > 8 and vol_spike > 2.0 and ice_roc_5 > -5:
            panic = True
            score = min(100, int(discount * 5 + vol_spike * 10))

        opp = "EXTREME_FEAR_ALL_IN" if discount > 15 else "PANIC_BUYING_ZONE" if discount > 8 and panic else "MILD_DISCOUNT" if discount > 3 else "FAIR_VALUE"

        return {"score": round(min(100, max(0, score)), 2), "panic_detected": panic, "discount_from_ma50_pct": round(discount, 2), "volume_spike_ratio": round(vol_spike, 2), "ice_roc_5d_pct": round(ice_roc_5, 2), "opportunity": opp}


# =============================================================================
# FUNDAMENTAL ANALYSIS
# =============================================================================

class FundamentalAnalysis:
    @staticmethod
    def calculate_safety_score(info: Dict, stock_df: pd.DataFrame) -> Dict:
        if info is None:
            return {"score": 0, "details": "NO_DATA"}

        score = 0
        details = {}

        eps_t = info.get("trailingEps")
        eps_f = info.get("forwardEps")
        if eps_t and eps_f and eps_t > 0:
            growth = ((eps_f - eps_t) / eps_t) * 100
            if growth > 10: score += 25
            elif growth > 0: score += 15
            elif growth > -10: score += 5
            details["eps_growth_pct"] = round(growth, 2)
        else:
            details["eps_growth_pct"] = None

        div_rate = info.get("dividendRate", 0)
        payout = info.get("payoutRatio")
        price = stock_df["Close"].iloc[-1] if stock_df is not None and not stock_df.empty else info.get("currentPrice", 0)

        if div_rate and div_rate > 0 and price > 0:
            yield_pct = (div_rate / price) * 100
            if yield_pct > 8: score += 25
            elif yield_pct > 5: score += 20
            elif yield_pct > 3: score += 10
            details["dividend_yield_pct"] = round(yield_pct, 2)
            details["dividend_rate"] = round(div_rate, 0)
        else:
            details["dividend_yield_pct"] = 0

        if payout and 0 < payout < 1:
            details["payout_ratio_pct"] = round(payout * 100, 2)
            if payout > 0.9: score = max(0, score - 10)
        else:
            details["payout_ratio_pct"] = None

        fcf = info.get("freeCashflow")
        if fcf and fcf > 0:
            score += 25
            details["fcf_positive"] = True
        else:
            details["fcf_positive"] = False

        debt = info.get("totalDebt", 0)
        equity = info.get("totalStockholderEquity", 0)
        if equity and equity > 0:
            der = debt / equity if debt else 0
            if der < 0.5: score += 25
            elif der < 1.0: score += 15
            elif der < 1.5: score += 5
            details["der"] = round(der, 2)
        else:
            details["der"] = None

        return {"score": round(max(0, min(100, score)), 2), "details": details}

    @staticmethod
    def calculate_forward_yield(info: Dict, price: float) -> Optional[float]:
        div = info.get("dividendRate")
        if div and price > 0:
            return round((div / price) * 100, 2)
        return None


# =============================================================================
# ENGINES
# =============================================================================

class SwingEngine:
    def __init__(self, ice_df):
        self.ice_df = ice_df
        self.ice_momentum = ICEAnalysis.calculate_momentum_score(ice_df) if ice_df is not None else {"score": 50, "trend": "UNKNOWN"}
        self.ice_current = ICEAnalysis.get_current_price(ice_df)

    def analyze_stock(self, ticker, stock_df):
        if stock_df is None or len(stock_df) < 30:
            return {"ticker": ticker, "mode": "SWING", "signal": "NO_DATA", "composite_score": 0}

        close = stock_df["Close"]
        volume = stock_df["Volume"]
        price = float(close.iloc[-1])

        # Technical
        ma10 = TechnicalIndicators.calculate_sma(close, SWING_MA_FAST).iloc[-1]
        ma20 = TechnicalIndicators.calculate_sma(close, SWING_MA_SLOW).iloc[-1]
        rsi = TechnicalIndicators.calculate_rsi(close, SWING_RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        _, _, macd_hist = TechnicalIndicators.calculate_macd(close)
        macd_val = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0

        tech_score = 0
        if price > ma10 > ma20: tech_score += 30
        elif price > ma10: tech_score += 20
        elif price > ma20: tech_score += 10
        if SWING_RSI_OVERSOLD < rsi_val < 60: tech_score += 35
        elif rsi_val <= SWING_RSI_OVERSOLD: tech_score += 25
        elif 60 <= rsi_val < SWING_RSI_OVERBOUGHT: tech_score += 20
        else: tech_score += 5
        if macd_val > 0: tech_score += 35
        elif macd_val > -0.5 * price / 100: tech_score += 15
        tech_score = max(0, min(100, tech_score))

        # Volume
        vol_ratio = TechnicalIndicators.calculate_volume_ratio(volume, 20)
        cur_ratio = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 1.0
        vol_score = 90 if cur_ratio > 2.0 else 70 if cur_ratio > 1.5 else 50 if cur_ratio > 1.2 else 30 if cur_ratio > 0.8 else 10
        if cur_ratio > 1.3 and close.iloc[-1] > close.iloc[-2]: vol_score = min(100, vol_score + 15)

        # Accumulation
        acc = TechnicalIndicators.detect_accumulation(stock_df)

        # Composite
        composite = self.ice_momentum["score"] * 0.40 + tech_score * 0.40 + vol_score * 0.20

        # Signal
        signal = "HOLD"
        tp, sl = 0, 0
        if composite >= 75 and rsi_val < SWING_RSI_OVERBOUGHT:
            signal = "BUY"
            if composite >= 85:
                tp = price * (1 + SWING_TP_MAX_PCT / 100)
                sl = price * (1 - SWING_SL_MIN_PCT / 100)
            else:
                tp = price * (1 + SWING_TP_MIN_PCT / 100)
                sl = price * (1 - SWING_SL_MAX_PCT / 100)
        elif composite < 40 and rsi_val > SWING_RSI_OVERBOUGHT:
            signal = "SELL"

        if signal == "BUY" and acc["accumulation_score"] >= 60:
            signal = "STRONG_BUY"

        if self.ice_current and self.ice_current < ICE_FLOOR_PRICE_USD and signal == "BUY":
            signal = "HOLD"

        return {
            "ticker": ticker, "mode": "SWING", "current_price": round(price, 0),
            "signal": signal, "composite_score": round(composite, 2),
            "target_profit": round(tp, 0), "stop_loss": round(sl, 0),
            "ice_filter_passed": self.ice_current is None or self.ice_current >= ICE_FLOOR_PRICE_USD,
            "ice_momentum": self.ice_momentum,
            "technical": {"score": round(tech_score, 2), "ma10": round(float(ma10), 0), "ma20": round(float(ma20), 0), "rsi": round(rsi_val, 2), "macd_histogram": round(macd_val, 2), "price_vs_ma10_pct": round(((price - ma10) / ma10) * 100, 2), "price_vs_ma20_pct": round(((price - ma20) / ma20) * 100, 2)},
            "volume_sentiment": {"score": round(min(100, vol_score), 2), "volume_ratio": round(cur_ratio, 2), "avg_volume_20d": int(volume.rolling(20).mean().iloc[-1]), "last_volume": int(volume.iloc[-1])},
            "accumulation": acc, "timestamp": datetime.now().isoformat(),
        }


class DividendEngine:
    def __init__(self, ice_df):
        self.ice_df = ice_df
        self.ice_trend = ICEAnalysis.calculate_trend_barrier_score(ice_df) if ice_df is not None else {"score": 50, "position": "NO_DATA"}
        self.ice_current = ICEAnalysis.get_current_price(ice_df)

    def analyze_stock(self, ticker, stock_df, info):
        if stock_df is None or len(stock_df) < 50:
            return {"ticker": ticker, "mode": "DIVIDEND", "signal": "NO_DATA", "composite_score": 0}

        close = stock_df["Close"]
        price = float(close.iloc[-1])

        fundamental = FundamentalAnalysis.calculate_safety_score(info, stock_df)
        fear_panic = ICEAnalysis.calculate_fear_panic_score(self.ice_df, stock_df)
        composite = fundamental["score"] * 0.40 + self.ice_trend["score"] * 0.40 + fear_panic["score"] * 0.20

        fwd_yield = FundamentalAnalysis.calculate_forward_yield(info, price)
        acc = TechnicalIndicators.detect_accumulation(stock_df)

        signal = "WAIT"
        if composite >= 75 and fear_panic.get("panic_detected") and fundamental["score"] >= 60:
            signal = "STRONG_ACCUMULATE"
        elif composite >= 65:
            signal = "ACCUMULATE"
        elif composite >= 45 and fwd_yield and fwd_yield >= 5:
            signal = "REINVEST_ZONE"
        if fwd_yield and fwd_yield >= DIV_TARGET_FORWARD_YIELD_PCT:
            signal = "SWITCH_TO_THIS_TICKER"

        ma50 = float(TechnicalIndicators.calculate_sma(close, DIV_MA_TREND).iloc[-1])
        ma200 = float(TechnicalIndicators.calculate_sma(close, DIV_MA_LONG).iloc[-1]) if len(close) >= 200 else None

        return {
            "ticker": ticker, "mode": "DIVIDEND", "current_price": round(price, 0),
            "signal": signal, "composite_score": round(composite, 2),
            "forward_yield_pct": fwd_yield, "ice_trend_barrier": self.ice_trend,
            "fundamental_safety": fundamental, "fear_panic_detector": fear_panic,
            "accumulation": acc, "ma50": round(ma50, 0),
            "ma200": round(ma200, 0) if ma200 else None,
            "price_vs_ma50_pct": round(((price - ma50) / ma50) * 100, 2),
            "price_vs_ma200_pct": round(((price - ma200) / ma200) * 100, 2) if ma200 else None,
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# MAIN
# =============================================================================

class CoalAlgoTrader:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.results = {
            "metadata": {"system": "Coal Algo-Trader v1.0", "generated_at": datetime.now().isoformat(), "ice_floor_filter_usd": ICE_FLOOR_PRICE_USD, "tickers_analyzed": COAL_TICKERS},
            "ice_newcastle": {}, "swing_mode": [], "dividend_mode": [], "alerts": [],
        }

    def run(self):
        logger.info("=" * 60)
        logger.info("COAL ALGO-TRADER v1.0 — Starting")
        logger.info("=" * 60)

        ice_df = self.fetcher.fetch_ice_newcastle()

        if ice_df is None:
            error_msg = "GAGAL MENGAMBIL DATA NEWCASTLE COAL."
            logger.critical(error_msg)
            self.results["ice_newcastle"] = {"status": "FAILED", "error_message": error_msg, "current_price_usd": None}
            self.results["alerts"].append({"type": "CRITICAL_DATA_FAILURE", "message": error_msg, "severity": "CRITICAL"})
            for t in COAL_TICKERS:
                self.results["swing_mode"].append({"ticker": t, "signal": "NO_DATA", "reason": "ICE Data Failed"})
                self.results["dividend_mode"].append({"ticker": t, "signal": "NO_DATA", "reason": "ICE Data Failed"})
            self._export()
            return

        logger.info("✅ Data Newcastle berhasil. Melanjutkan...")
        ice_current = ICEAnalysis.get_current_price(ice_df)
        self.results["ice_newcastle"] = {
            "status": "SUCCESS", "current_price_usd": round(ice_current, 2),
            "floor_threshold_usd": ICE_FLOOR_PRICE_USD, "above_floor": ice_current >= ICE_FLOOR_PRICE_USD,
            "momentum": ICEAnalysis.calculate_momentum_score(ice_df),
            "trend_barrier": ICEAnalysis.calculate_trend_barrier_score(ice_df),
        }

        if ice_current < ICE_FLOOR_PRICE_USD:
            self.results["alerts"].append({"type": "ICE_FLOOR_BREACH", "message": f"⚠️ ICE DI BAWAH FLOOR: ${ice_current:.2f}/ton", "severity": "HIGH"})

        swing_engine = SwingEngine(ice_df)
        dividend_engine = DividendEngine(ice_df)

        for ticker in COAL_TICKERS:
            logger.info(f"\nAnalyzing {ticker}...")
            stock_df = self.fetcher.fetch_stock_history(ticker)
            info = self.fetcher.fetch_stock_info(ticker)

            sr = swing_engine.analyze_stock(ticker, stock_df)
            self.results["swing_mode"].append(sr)
            dr = dividend_engine.analyze_stock(ticker, stock_df, info)
            self.results["dividend_mode"].append(dr)

            if sr["signal"] in ["STRONG_BUY", "BUY"]:
                self.results["alerts"].append({"type": "SWING_BUY_SIGNAL", "ticker": ticker, "price": sr.get("current_price"), "tp": sr.get("target_profit"), "sl": sr.get("stop_loss"), "severity": "HIGH" if sr["signal"] == "STRONG_BUY" else "MEDIUM"})
            if dr["signal"] in ["STRONG_ACCUMULATE", "SWITCH_TO_THIS_TICKER"]:
                self.results["alerts"].append({"type": "DIVIDEND_ACCUMULATE_SIGNAL", "ticker": ticker, "price": dr.get("current_price"), "forward_yield": dr.get("forward_yield_pct"), "severity": "HIGH"})
            acc = sr.get("accumulation", {})
            if acc.get("signal") == "STRONG_ACCUMULATION":
                self.results["alerts"].append({"type": "BANDAR_ACCUMULATION_DETECTED", "ticker": ticker, "score": acc.get("accumulation_score"), "severity": "HIGH"})

        self._export()

    def _export(self):
        self.results["swing_mode"].sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        self.results["dividend_mode"].sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        self.results["summary"] = {
            "total_tickers": len(COAL_TICKERS),
            "swing_buy_signals": sum(1 for r in self.results["swing_mode"] if r.get("signal") in ["BUY", "STRONG_BUY"]),
            "dividend_accumulate_signals": sum(1 for r in self.results["dividend_mode"] if r.get("signal") in ["ACCUMULATE", "STRONG_ACCUMULATE"]),
            "total_alerts": len(self.results["alerts"]),
            "ice_above_floor": self.results["ice_newcastle"].get("above_floor", False),
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"\n✅ EXPORTED: {OUTPUT_FILE}")
        logger.info(f"   Swing BUY: {self.results['summary']['swing_buy_signals']}")
        logger.info(f"   Div ACCUMULATE: {self.results['summary']['dividend_accumulate_signals']}")


if __name__ == "__main__":
    CoalAlgoTrader().run()
