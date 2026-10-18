"""
=============================================================================
COAL ALGO-TRADER v1.0 — Dual-Engine Recommendation System
=============================================================================
Engine 1: Swing Trading Mode (Capital Gain Hunter)
Engine 2: Dividend Investing Mode (Yield Accumulator)

Data Source : yfinance (ICE Newcastle ^NCF=F + IDX Coal Stocks)
Scheduler   : GitHub Actions Cron (16:30 WIB)
Output      : coal_data.json → Static Frontend (GitHub Pages)
Author      : Custom Build for ITMG Dividend Collector Profile
=============================================================================
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("CoalAlgoTrader")

# Ticker Universe
COAL_TICKERS = ["ITMG.JK", "PTBA.JK", "ADRO.JK", "PTCO.JK", "BUMI.JK", "MBAP.JK"]
ICE_TICKER = "^NCF=F"  # ICE Newcastle Coal Futures

# Filter Thresholds
ICE_FLOOR_PRICE_USD = 120.0  # Minimum ICE Newcastle price untuk aktifkan sinyal BUY

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
DIV_MAX_DER = 1.0
DIV_MIN_PAYOUT_HISTORY_YEARS = 3
DIV_TARGET_FORWARD_YIELD_PCT = 12.0  # Untuk switch ITMG → PTBA

# Weight Configurations
SWING_WEIGHTS = {
    "ice_momentum": 0.40,
    "technical": 0.40,
    "volume_sentiment": 0.20,
}

DIVIDEND_WEIGHTS = {
    "fundamental_safety": 0.40,
    "ice_trend_barrier": 0.40,
    "fear_panic_detector": 0.20,
}

# Output
OUTPUT_FILE = "coal_data.json"


# =============================================================================
# DATA FETCHING MODULE
# =============================================================================

class DataFetcher:
    """Mengambil data harga dan fundamental dari yfinance."""

    @staticmethod
    def fetch_ice_newcastle(period: str = "6mo") -> Optional[pd.DataFrame]:
        """Ambil data ICE Newcastle Futures (^NCF=F)."""
        logger.info(f"Fetching ICE Newcastle data ({period})...")
        try:
            ice = yf.Ticker(ICE_TICKER)
            df = ice.history(period=period)
            if df.empty:
                logger.warning("ICE Newcastle data kosong. Coba ticker alternatif.")
                # Fallback: coba ticker alternatif jika ^NCF=F bermasalah
                ice_alt = yf.Ticker("NC=F")
                df = ice_alt.history(period=period)
            if df.empty:
                logger.error("Gagal mengambil data ICE Newcastle.")
                return None
            logger.info(f"ICE Newcastle: {len(df)} baris data diperoleh.")
            return df
        except Exception as e:
            logger.error(f"Error fetching ICE data: {e}")
            return None

    @staticmethod
    def fetch_stock_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Ambil data harga historis saham."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                logger.warning(f"{ticker}: Data harga kosong.")
                return None
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker} history: {e}")
            return None

    @staticmethod
    def fetch_stock_info(ticker: str) -> Optional[Dict]:
        """Ambil data fundamental saham."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or len(info) < 5:
                logger.warning(f"{ticker}: Data info fundamental tidak lengkap.")
                return None
            return info
        except Exception as e:
            logger.error(f"Error fetching {ticker} info: {e}")
            return None


# =============================================================================
# TECHNICAL INDICATORS MODULE
# =============================================================================

class TechnicalIndicators:
    """Kalkulasi indikator teknikal."""

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # Gunakan EMA setelah rolling awal untuk Wilder's smoothing
        avg_gain = avg_gain.iloc[period:]
        avg_loss = avg_loss.iloc[period:]
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_rate_of_change(series: pd.Series, period: int = 14) -> pd.Series:
        """Rate of Change (ROC) dalam persentase."""
        roc = ((series - series.shift(period)) / series.shift(period)) * 100
        return roc

    @staticmethod
    def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Rasio volume hari ini vs rata-rata volume N hari."""
        avg_vol = volume.rolling(window=period, min_periods=period).mean()
        ratio = volume / avg_vol.replace(0, np.nan)
        return ratio

    @staticmethod
    def detect_accumulation(df: pd.DataFrame, lookback: int = 10) -> Dict:
        """
        Deteksi akumulasi bandar sederhana:
        - Volume spike (>1.5x avg) + harga naik tipis/stabil (bukan pump)
        - On-Balance Volume (OBV) trending up
        - Close > Open konsisten di hari volume tinggi
        """
        if df is None or len(df) < lookback + 5:
            return {"accumulation_score": 0, "signal": "NO_DATA"}

        recent = df.tail(lookback)
        vol_ratio = recent["Volume"] / recent["Volume"].rolling(20).mean()
        vol_ratio = vol_ratio.fillna(1.0)

        # OBV Calculation
        obv = [0]
        for i in range(1, len(recent)):
            if recent["Close"].iloc[i] > recent["Close"].iloc[i - 1]:
                obv.append(obv[-1] + recent["Volume"].iloc[i])
            elif recent["Close"].iloc[i] < recent["Close"].iloc[i - 1]:
                obv.append(obv[-1] - recent["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        obv_series = pd.Series(obv, index=recent.index)

        # Scoring akumulasi
        score = 0
        signals = []

        # 1. Volume spike tanpa harga melonjak (price change < 3% tapi vol > 1.5x)
        price_change_pct = ((recent["Close"].iloc[-1] - recent["Close"].iloc[0]) / recent["Close"].iloc[0]) * 100
        high_vol_days = (vol_ratio > 1.5).sum()

        if high_vol_days >= 3 and abs(price_change_pct) < 5:
            score += 40
            signals.append("VOLUME_SPIKE_STABLE_PRICE")

        # 2. OBV trending up
        if len(obv_series) >= 5:
            obv_trend = obv_series.iloc[-1] - obv_series.iloc[-5]
            if obv_trend > 0:
                score += 30
                signals.append("OBV_UPTREND")

        # 3. Close > Open di hari volume tinggi (buying pressure)
        high_vol_mask = vol_ratio > 1.3
        if high_vol_mask.any():
            buy_candle_ratio = (recent.loc[high_vol_mask, "Close"] > recent.loc[high_vol_mask, "Open"]).mean()
            if buy_candle_ratio > 0.6:
                score += 30
                signals.append("BUYING_PRESSURE_HIGH_VOL")

        # Determine signal
        if score >= 70:
            signal = "STRONG_ACCUMULATION"
        elif score >= 40:
            signal = "MODERATE_ACCUMULATION"
        else:
            signal = "NO_SIGNIFICANT_ACCUMULATION"

        return {
            "accumulation_score": score,
            "signal": signal,
            "signals_detail": signals,
            "high_vol_days": int(high_vol_days),
            "price_change_pct": round(price_change_pct, 2),
        }


# =============================================================================
# ICE NEWCASTLE ANALYSIS MODULE
# =============================================================================

class ICEAnalysis:
    """Analisis ICE Newcastle Futures untuk kedua mode."""

    @staticmethod
    def get_current_price(ice_df: pd.DataFrame) -> Optional[float]:
        if ice_df is None or ice_df.empty:
            return None
        return float(ice_df["Close"].iloc[-1])

    @staticmethod
    def calculate_momentum_score(ice_df: pd.DataFrame) -> Dict:
        """
        SWING MODE: ICE Momentum Score (Bobot 40%)
        Menggunakan ROC 5, 10, 20 hari dengan pembobotan.
        """
        if ice_df is None or len(ice_df) < 25:
            return {"score": 0, "roc_5d": 0, "roc_10d": 0, "roc_20d": 0, "trend": "UNKNOWN"}

        close = ice_df["Close"]
        roc_5 = TechnicalIndicators.calculate_rate_of_change(close, 5).iloc[-1]
        roc_10 = TechnicalIndicators.calculate_rate_of_change(close, 10).iloc[-1]
        roc_20 = TechnicalIndicators.calculate_rate_of_change(close, 20).iloc[-1]

        # Normalisasi ke skor 0-100
        # ROC positif besar = skor tinggi, ROC negatif = skor rendah
        def roc_to_score(roc_val):
            # Clamp antara -15% dan +15%
            clamped = max(-15, min(15, roc_val))
            return (clamped + 15) / 30 * 100

        score_5 = roc_to_score(roc_5)
        score_10 = roc_to_score(roc_10)
        score_20 = roc_to_score(roc_20)

        # Bobot: ROC5 (50%), ROC10 (30%), ROC20 (20%)
        composite = score_5 * 0.50 + score_10 * 0.30 + score_20 * 0.20

        # Trend determination
        ma_20 = close.rolling(20).mean().iloc[-1]
        current = close.iloc[-1]
        if current > ma_20 * 1.02:
            trend = "BULLISH"
        elif current < ma_20 * 0.98:
            trend = "BEARISH"
        else:
            trend = "SIDEWAYS"

        return {
            "score": round(composite, 2),
            "roc_5d": round(roc_5, 2),
            "roc_10d": round(roc_10, 2),
            "roc_20d": round(roc_20, 2),
            "trend": trend,
            "current_price_usd": round(current, 2),
        }

    @staticmethod
    def calculate_trend_barrier_score(ice_df: pd.DataFrame) -> Dict:
        """
        DIVIDEND MODE: ICE Trend Barrier Score (Bobot 40%)
        Posisi harga ICE terhadap MA50 dan MA200.
        """
        if ice_df is None or len(ice_df) < 200:
            # Fallback jika data < 200 hari
            if ice_df is not None and len(ice_df) >= 50:
                close = ice_df["Close"]
                ma_50 = close.rolling(50).mean().iloc[-1]
                current = close.iloc[-1]
                above_ma50 = current > ma_50
                score = 70 if above_ma50 else 30
                return {
                    "score": score,
                    "above_ma50": above_ma50,
                    "above_ma200": None,
                    "position": "ABOVE_MA50" if above_ma50 else "BELOW_MA50",
                    "current_price_usd": round(current, 2),
                }
            return {"score": 0, "above_ma50": None, "above_ma200": None, "position": "NO_DATA", "current_price_usd": None}

        close = ice_df["Close"]
        ma_50 = close.rolling(50).mean().iloc[-1]
        ma_200 = close.rolling(200).mean().iloc[-1]
        current = close.iloc[-1]

        above_ma50 = current > ma_50
        above_ma200 = current > ma_200

        # Scoring
        score = 0
        if above_ma200:
            score += 50
        if above_ma50:
            score += 30
        # Bonus jika golden cross (MA50 > MA200)
        if ma_50 > ma_200:
            score += 20
        # Penalty jika death cross
        elif ma_50 < ma_200:
            score -= 10

        score = max(0, min(100, score))

        position = "STRONG_BULL" if (above_ma50 and above_ma200) else \
                   "WEAK_BULL" if above_ma200 else \
                   "RECOVERY" if above_ma50 else "BEARISH"

        return {
            "score": round(score, 2),
            "above_ma50": above_ma50,
            "above_ma200": above_ma200,
            "position": position,
            "ma50": round(float(ma_50), 2),
            "ma200": round(float(ma_200), 2),
            "current_price_usd": round(float(current), 2),
        }

    @staticmethod
    def calculate_fear_panic_score(ice_df: pd.DataFrame, stock_df: pd.DataFrame) -> Dict:
        """
        DIVIDEND MODE: Fear/Panic Detector (Bobot 20%)
        Mendeteksi panic selling / mispricing opportunity.
        Indikator:
        - Harga saham jauh di bawah MA50 (>10% discount)
        - Volume spike drastis (>3x average)
        - ICE masih stabil (tidak crash)
        """
        if stock_df is None or len(stock_df) < 50 or ice_df is None or len(ice_df) < 20:
            return {"score": 0, "panic_detected": False, "opportunity": "NO_DATA"}

        close = stock_df["Close"]
        volume = stock_df["Volume"]
        ma_50_stock = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]

        # Discount dari MA50
        discount_from_ma50 = ((ma_50_stock - current_price) / ma_50_stock) * 100

        # Volume spike
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        vol_spike_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1

        # ICE stability (ROC 5 hari ICE tidak crash > -10%)
        ice_close = ice_df["Close"]
        ice_roc_5 = ((ice_close.iloc[-1] - ice_close.iloc[-5]) / ice_close.iloc[-5]) * 100 if len(ice_close) >= 5 else 0

        score = 0
        panic_detected = False

        # Panic selling terdeteksi jika:
        # 1. Discount dari MA50 > 8%
        # 2. Volume spike > 2x
        # 3. ICE tidak ikut crash (masih > -$5% ROC)
        if discount_from_ma50 > 8 and vol_spike_ratio > 2.0 and ice_roc_5 > -5:
            panic_detected = True
            score = min(100, int(discount_from_ma50 * 5 + vol_spike_ratio * 10))

        # Opportunity zone
        if discount_from_ma50 > 15:
            opportunity = "EXTREME_FEAR_ALL_IN"
        elif discount_from_ma50 > 8 and panic_detected:
            opportunity = "PANIC_BUYING_ZONE"
        elif discount_from_ma50 > 3:
            opportunity = "MILD_DISCOUNT"
        else:
            opportunity = "FAIR_VALUE"

        return {
            "score": round(min(100, max(0, score)), 2),
            "panic_detected": panic_detected,
            "discount_from_ma50_pct": round(discount_from_ma50, 2),
            "volume_spike_ratio": round(vol_spike_ratio, 2),
            "ice_roc_5d_pct": round(ice_roc_5, 2),
            "opportunity": opportunity,
        }


# =============================================================================
# FUNDAMENTAL ANALYSIS MODULE
# =============================================================================

class FundamentalAnalysis:
    """Analisis fundamental untuk Dividend Mode."""

    @staticmethod
    def calculate_safety_score(info: Dict, stock_df: pd.DataFrame) -> Dict:
        """
        DIVIDEND MODE: Fundamental Safety Score (Bobot 40%)
        Komponen:
        - EPS Growth (quarterly) : 25%
        - Dividend Payout Consistency : 25%
        - Free Cash Flow positivity : 25%
        - DER < 100% : 25%
        """
        if info is None:
            return {"score": 0, "details": "NO_DATA"}

        score = 0
        details = {}

        # 1. EPS Growth (25 poin max)
        eps_trailing = info.get("trailingEps", None)
        eps_forward = info.get("forwardEps", None)
        if eps_trailing and eps_forward and eps_trailing > 0:
            eps_growth = ((eps_forward - eps_trailing) / eps_trailing) * 100
            if eps_growth > 10:
                score += 25
            elif eps_growth > 0:
                score += 15
            elif eps_growth > -10:
                score += 5
            details["eps_growth_pct"] = round(eps_growth, 2)
        else:
            details["eps_growth_pct"] = None

        # 2. Dividend Payout Consistency (25 poin max)
        div_rate = info.get("dividendRate", 0)
        div_yield = info.get("dividendYield", 0)
        payout_ratio = info.get("payoutRatio", None)
        current_price = stock_df["Close"].iloc[-1] if stock_df is not None and not stock_df.empty else info.get("currentPrice", 0)

        if div_rate and div_rate > 0 and current_price > 0:
            actual_yield = (div_rate / current_price) * 100
            if actual_yield > 8:
                score += 25
            elif actual_yield > 5:
                score += 20
            elif actual_yield > 3:
                score += 10
            details["dividend_yield_pct"] = round(actual_yield, 2)
            details["dividend_rate"] = round(div_rate, 0)
        else:
            details["dividend_yield_pct"] = 0

        if payout_ratio and 0 < payout_ratio < 1:
            details["payout_ratio_pct"] = round(payout_ratio * 100, 2)
            if payout_ratio < 0.8:
                score += 0  # Sudah dihitung di yield
            # Payout ratio sehat < 90%
            if payout_ratio > 0.9:
                score = max(0, score - 10)  # Penalty payout terlalu tinggi
        else:
            details["payout_ratio_pct"] = None

        # 3. Free Cash Flow (25 poin max)
        fcf = info.get("freeCashflow", None)
        if fcf and fcf > 0:
            score += 25
            details["fcf_positive"] = True
            details["fcf_value"] = fcf
        else:
            details["fcf_positive"] = False
            details["fcf_value"] = fcf

        # 4. DER < 100% (25 poin max)
        total_debt = info.get("totalDebt", 0)
        total_equity = info.get("totalStockholderEquity", 0)
        if total_equity and total_equity > 0:
            der = total_debt / total_equity if total_debt else 0
            if der < 0.5:
                score += 25
            elif der < 1.0:
                score += 15
            elif der < 1.5:
                score += 5
            details["der"] = round(der, 2)
        else:
            details["der"] = None

        score = max(0, min(100, score))

        return {
            "score": round(score, 2),
            "details": details,
        }

    @staticmethod
    def calculate_forward_yield(info: Dict, current_price: float) -> Optional[float]:
        """Hitung Forward Dividend Yield."""
        div_rate = info.get("dividendRate", None)
        if div_rate and current_price > 0:
            return round((div_rate / current_price) * 100, 2)
        return None


# =============================================================================
# SWING TRADING ENGINE
# =============================================================================

class SwingEngine:
    """
    ENGINE 1: Swing Trading Mode (Capital Gain Hunter)
    Horizon: 3 hari - 4 minggu
    """

    def __init__(self, ice_df: pd.DataFrame):
        self.ice_df = ice_df
        self.ice_momentum = ICEAnalysis.calculate_momentum_score(ice_df)
        self.ice_current = ICEAnalysis.get_current_price(ice_df)

    def analyze_stock(self, ticker: str, stock_df: pd.DataFrame) -> Dict:
        """Analisis lengkap per saham untuk Swing Mode."""
        if stock_df is None or len(stock_df) < 30:
            return self._empty_result(ticker, "INSUFFICIENT_DATA")

        close = stock_df["Close"]
        volume = stock_df["Volume"]
        current_price = float(close.iloc[-1])

        # --- Technical Score (40%) ---
        tech_score = self._calculate_technical_score(close)

        # --- Volume/Sentiment Score (20%) ---
        vol_score = self._calculate_volume_score(volume, close)

        # --- Accumulation Detection ---
        accumulation = TechnicalIndicators.detect_accumulation(stock_df)

        # --- Composite Score ---
        ice_score = self.ice_momentum["score"]
        composite = (
            ice_score * SWING_WEIGHTS["ice_momentum"] +
            tech_score["score"] * SWING_WEIGHTS["technical"] +
            vol_score["score"] * SWING_WEIGHTS["volume_sentiment"]
        )

        # --- Signal Generation ---
        signal, tp, sl = self._generate_signal(
            composite, tech_score, current_price, close, accumulation
        )

        # --- ICE Floor Filter ---
        ice_filter_passed = True
        if self.ice_current and self.ice_current < ICE_FLOOR_PRICE_USD:
            ice_filter_passed = False
            if signal == "BUY":
                signal = "HOLD"  # Downgrade BUY ke HOLD jika ICE < $120

        return {
            "ticker": ticker,
            "mode": "SWING",
            "current_price": round(current_price, 0),
            "signal": signal,
            "composite_score": round(composite, 2),
            "target_profit": tp,
            "stop_loss": sl,
            "ice_filter_passed": ice_filter_passed,
            "ice_momentum": self.ice_momentum,
            "technical": tech_score,
            "volume_sentiment": vol_score,
            "accumulation": accumulation,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_technical_score(self, close: pd.Series) -> Dict:
        """Hitung skor teknikal (MA, RSI, MACD)."""
        current = float(close.iloc[-1])
        ma10 = TechnicalIndicators.calculate_sma(close, SWING_MA_FAST).iloc[-1]
        ma20 = TechnicalIndicators.calculate_sma(close, SWING_MA_SLOW).iloc[-1]
        rsi = TechnicalIndicators.calculate_rsi(close, SWING_RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(close)
        macd_val = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0

        score = 0

        # MA Alignment (30 poin)
        if current > ma10 > ma20:
            score += 30  # Bullish alignment sempurna
        elif current > ma10:
            score += 20
        elif current > ma20:
            score += 10

        # RSI (35 poin)
        if SWING_RSI_OVERSOLD < rsi_val < 60:
            score += 35  # Zone optimal
        elif rsi_val <= SWING_RSI_OVERSOLD:
            score += 25  # Oversold = potensi bounce
        elif 60 <= rsi_val < SWING_RSI_OVERBOUGHT:
            score += 20
        else:
            score += 5  # Overbought

        # MACD (35 poin)
        if macd_val > 0:
            score += 35
        elif macd_val > -0.5 * float(close.iloc[-1]) / 100:
            score += 15
        else:
            score += 0

        score = max(0, min(100, score))

        return {
            "score": round(score, 2),
            "ma10": round(float(ma10), 0),
            "ma20": round(float(ma20), 0),
            "rsi": round(rsi_val, 2),
            "macd_histogram": round(macd_val, 2),
            "price_vs_ma10_pct": round(((current - ma10) / ma10) * 100, 2),
            "price_vs_ma20_pct": round(((current - ma20) / ma20) * 100, 2),
        }

    def _calculate_volume_score(self, volume: pd.Series, close: pd.Series) -> Dict:
        """Hitung skor volume/sentimen."""
        vol_ratio = TechnicalIndicators.calculate_volume_ratio(volume, 20)
        current_ratio = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 1.0

        score = 0
        if current_ratio > 2.0:
            score = 90  # Volume explosion
        elif current_ratio > 1.5:
            score = 70
        elif current_ratio > 1.2:
            score = 50
        elif current_ratio > 0.8:
            score = 30
        else:
            score = 10

        # Bonus jika volume naik + harga naik
        if current_ratio > 1.3 and close.iloc[-1] > close.iloc[-2]:
            score = min(100, score + 15)

        return {
            "score": round(min(100, score), 2),
            "volume_ratio": round(current_ratio, 2),
            "avg_volume_20d": int(volume.rolling(20).mean().iloc[-1]),
            "last_volume": int(volume.iloc[-1]),
        }

    def _generate_signal(self, composite: float, tech: Dict, price: float,
                         close: pd.Series, accumulation: Dict) -> Tuple[str, float, float]:
        """Generate sinyal BUY/HOLD/SELL + TP/SL."""
        tp = 0
        sl = 0

        if composite >= 75 and tech["rsi"] < SWING_RSI_OVERBOUGHT:
            signal = "BUY"
            # TP berdasarkan volatilitas
            if composite >= 85:
                tp = price * (1 + SWING_TP_MAX_PCT / 100)
                sl = price * (1 - SWING_SL_MIN_PCT / 100)
            else:
                tp = price * (1 + SWING_TP_MIN_PCT / 100)
                sl = price * (1 - SWING_SL_MAX_PCT / 100)
        elif composite >= 55:
            signal = "HOLD"
        elif composite < 40 and tech["rsi"] > SWING_RSI_OVERBOUGHT:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Upgrade ke STRONG BUY jika ada akumulasi bandar
        if signal == "BUY" and accumulation["accumulation_score"] >= 60:
            signal = "STRONG_BUY"

        return signal, round(tp, 0), round(sl, 0)

    def _empty_result(self, ticker: str, reason: str) -> Dict:
        return {
            "ticker": ticker,
            "mode": "SWING",
            "signal": "NO_DATA",
            "reason": reason,
            "composite_score": 0,
        }


# =============================================================================
# DIVIDEND INVESTING ENGINE
# =============================================================================

class DividendEngine:
    """
    ENGINE 2: Dividend Investing Mode (Yield Accumulator)
    Horizon: Jangka panjang (Tahunan)
    """

    def __init__(self, ice_df: pd.DataFrame):
        self.ice_df = ice_df
        self.ice_trend = ICEAnalysis.calculate_trend_barrier_score(ice_df)
        self.ice_current = ICEAnalysis.get_current_price(ice_df)

    def analyze_stock(self, ticker: str, stock_df: pd.DataFrame, info: Dict) -> Dict:
        """Analisis lengkap per saham untuk Dividend Mode."""
        if stock_df is None or len(stock_df) < 50:
            return self._empty_result(ticker, "INSUFFICIENT_DATA")

        close = stock_df["Close"]
        current_price = float(close.iloc[-1])

        # --- Fundamental Safety Score (40%) ---
        fundamental = FundamentalAnalysis.calculate_safety_score(info, stock_df)

        # --- ICE Trend Barrier Score (40%) ---
        ice_score = self.ice_trend["score"]

        # --- Fear/Panic Detector (20%) ---
        fear_panic = ICEAnalysis.calculate_fear_panic_score(self.ice_df, stock_df)

        # --- Composite Score ---
        composite = (
            fundamental["score"] * DIVIDEND_WEIGHTS["fundamental_safety"] +
            ice_score * DIVIDEND_WEIGHTS["ice_trend_barrier"] +
            fear_panic["score"] * DIVIDEND_WEIGHTS["fear_panic_detector"]
        )

        # --- Forward Yield Calculation ---
        forward_yield = FundamentalAnalysis.calculate_forward_yield(info, current_price)

        # --- Signal Generation ---
        signal = self._generate_signal(composite, fear_panic, forward_yield, fundamental)

        # --- Accumulation Detection ---
        accumulation = TechnicalIndicators.detect_accumulation(stock_df)

        # --- MA Position ---
        ma50 = float(TechnicalIndicators.calculate_sma(close, DIV_MA_TREND).iloc[-1])
        ma200 = float(TechnicalIndicators.calculate_sma(close, DIV_MA_LONG).iloc[-1]) if len(close) >= 200 else None

        return {
            "ticker": ticker,
            "mode": "DIVIDEND",
            "current_price": round(current_price, 0),
            "signal": signal,
            "composite_score": round(composite, 2),
            "forward_yield_pct": forward_yield,
            "ice_trend_barrier": self.ice_trend,
            "fundamental_safety": fundamental,
            "fear_panic_detector": fear_panic,
            "accumulation": accumulation,
            "ma50": round(ma50, 0),
            "ma200": round(ma200, 0) if ma200 else None,
            "price_vs_ma50_pct": round(((current_price - ma50) / ma50) * 100, 2),
            "price_vs_ma200_pct": round(((current_price - ma200) / ma200) * 100, 2) if ma200 else None,
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_signal(self, composite: float, fear_panic: Dict,
                         forward_yield: Optional[float], fundamental: Dict) -> str:
        """Generate sinyal untuk Dividend Mode."""

        # STRONG ACCUMULATE: Skor tinggi + panic selling + fundamental kuat
        if (composite >= 75 and
            fear_panic.get("panic_detected", False) and
            fundamental["score"] >= 60):
            return "STRONG_ACCUMULATE"

        # ACCUMULATE: Skor cukup tinggi atau forward yield menarik
        if composite >= 65:
            return "ACCUMULATE"

        # REINVEST ZONE: DCA dividen biasa
        if composite >= 45 and forward_yield and forward_yield >= 5:
            return "REINVEST_ZONE"

        # PTBA Switch Check: Jika PTBA forward yield > 12%
        if forward_yield and forward_yield >= DIV_TARGET_FORWARD_YIELD_PCT:
            return "SWITCH_TO_THIS_TICKER"

        # WAIT
        return "WAIT"

    def _empty_result(self, ticker: str, reason: str) -> Dict:
        return {
            "ticker": ticker,
            "mode": "DIVIDEND",
            "signal": "NO_DATA",
            "reason": reason,
            "composite_score": 0,
        }


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class CoalAlgoTrader:
    """Orchestrator utama yang menjalankan kedua engine dan mengekspor hasil."""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.results = {
            "metadata": {
                "system": "Coal Algo-Trader v1.0",
                "generated_at": datetime.now().isoformat(),
                "ice_floor_filter_usd": ICE_FLOOR_PRICE_USD,
                "tickers_analyzed": COAL_TICKERS,
            },
            "ice_newcastle": {},
            "swing_mode": [],
            "dividend_mode": [],
            "alerts": [],
        }

    def run(self):
        """Jalankan seluruh pipeline analisis."""
        logger.info("=" * 60)
        logger.info("COAL ALGO-TRADER v1.0 — Starting Analysis Pipeline")
        logger.info("=" * 60)

        # Step 1: Fetch ICE Newcastle Data
        ice_df = self.fetcher.fetch_ice_newcastle()
        if ice_df is None:
            logger.critical("CRITICAL: Tidak bisa mengambil data ICE Newcastle. Abort.")
            self._export_empty("ICE_DATA_UNAVAILABLE")
            return

        ice_current = ICEAnalysis.get_current_price(ice_df)
        self.results["ice_newcastle"] = {
            "current_price_usd": round(ice_current, 2) if ice_current else None,
            "floor_threshold_usd": ICE_FLOOR_PRICE_USD,
            "above_floor": ice_current >= ICE_FLOOR_PRICE_USD if ice_current else False,
            "momentum": ICEAnalysis.calculate_momentum_score(ice_df),
            "trend_barrier": ICEAnalysis.calculate_trend_barrier_score(ice_df),
        }

        # Step 2: Alert jika ICE di bawah floor
        if ice_current and ice_current < ICE_FLOOR_PRICE_USD:
            alert_msg = (
                f"⚠️ ICE NEWCASTLE DI BAWAH FLOOR: ${ice_current:.2f}/ton "
                f"(Floor: ${ICE_FLOOR_PRICE_USD}/ton). "
                f"Semua sinyal BUY Swing Mode di-DOWNGRADE ke HOLD."
            )
            logger.warning(alert_msg)
            self.results["alerts"].append({
                "type": "ICE_FLOOR_BREACH",
                "message": alert_msg,
                "severity": "HIGH",
            })

        # Step 3: Initialize Engines
        swing_engine = SwingEngine(ice_df)
        dividend_engine = DividendEngine(ice_df)

        # Step 4: Analyze Each Ticker
        for ticker in COAL_TICKERS:
            logger.info(f"Analyzing {ticker}...")

            stock_df = self.fetcher.fetch_stock_history(ticker)
            info = self.fetcher.fetch_stock_info(ticker)

            # Swing Analysis
            swing_result = swing_engine.analyze_stock(ticker, stock_df)
            self.results["swing_mode"].append(swing_result)

            # Dividend Analysis
            div_result = dividend_engine.analyze_stock(ticker, stock_df, info)
            self.results["dividend_mode"].append(div_result)

            # Generate alerts for strong signals
            if swing_result["signal"] in ["STRONG_BUY", "BUY"]:
                self.results["alerts"].append({
                    "type": "SWING_BUY_SIGNAL",
                    "ticker": ticker,
                    "price": swing_result.get("current_price"),
                    "tp": swing_result.get("target_profit"),
                    "sl": swing_result.get("stop_loss"),
                    "score": swing_result.get("composite_score"),
                    "severity": "MEDIUM" if swing_result["signal"] == "BUY" else "HIGH",
                })

            if div_result["signal"] in ["STRONG_ACCUMULATE", "SWITCH_TO_THIS_TICKER"]:
                self.results["alerts"].append({
                    "type": "DIVIDEND_ACCUMULATE_SIGNAL",
                    "ticker": ticker,
                    "price": div_result.get("current_price"),
                    "forward_yield": div_result.get("forward_yield_pct"),
                    "score": div_result.get("composite_score"),
                    "severity": "HIGH",
                })

            # Accumulation alert
            acc = swing_result.get("accumulation", {})
            if acc.get("signal") == "STRONG_ACCUMULATION":
                self.results["alerts"].append({
                    "type": "BANDAR_ACCUMULATION_DETECTED",
                    "ticker": ticker,
                    "score": acc.get("accumulation_score"),
                    "details": acc.get("signals_detail", []),
                    "severity": "HIGH",
                })

        # Step 5: Export
        self._export()

    def _export(self):
        """Ekspor hasil ke JSON."""
        # Sort by composite score descending
        self.results["swing_mode"].sort(
            key=lambda x: x.get("composite_score", 0), reverse=True
        )
        self.results["dividend_mode"].sort(
            key=lambda x: x.get("composite_score", 0), reverse=True
        )

        # Add summary
        self.results["summary"] = {
            "total_tickers": len(COAL_TICKERS),
            "swing_buy_signals": sum(
                1 for r in self.results["swing_mode"]
                if r.get("signal") in ["BUY", "STRONG_BUY"]
            ),
            "dividend_accumulate_signals": sum(
                1 for r in self.results["dividend_mode"]
                if r.get("signal") in ["ACCUMULATE", "STRONG_ACCUMULATE", "SWITCH_TO_THIS_TICKER"]
            ),
            "total_alerts": len(self.results["alerts"]),
            "ice_above_floor": self.results["ice_newcastle"].get("above_floor", False),
        }

        with open(OUTPUT_FILE, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"✅ Hasil analisis berhasil diekspor ke {OUTPUT_FILE}")
        logger.info(f"   Swing BUY signals: {self.results['summary']['swing_buy_signals']}")
        logger.info(f"   Dividend ACCUMULATE signals: {self.results['summary']['dividend_accumulate_signals']}")
        logger.info(f"   Total alerts: {self.results['summary']['total_alerts']}")

    def _export_empty(self, reason: str):
        """Ekspor JSON kosong jika error critical."""
        self.results["metadata"]["error"] = reason
        self.results["metadata"]["generated_at"] = datetime.now().isoformat()
        with open(OUTPUT_FILE, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.error(f"Exported empty result: {reason}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    trader = CoalAlgoTrader()
    trader.run()
