#!/usr/bin/env python3
"""
bn_suggester.py
Single-file Bank Nifty trade suggestion tool.

Usage:
1. Create a file named `key` in the same folder. Format (simple INI-like):
   MODE=mock          # or "kite" for Zerodha KiteConnect
   KITE_API_KEY=...   # only if MODE=kite
   KITE_ACCESS_TOKEN=... # only if MODE=kite
   PREFERRED_EXPIRY_DAYS=7  # default preference in days for expiry selection

2. Install dependencies for live Kite mode:
   pip install kiteconnect pandas numpy

3. Run:
   python bn_suggester.py

Output: JSON-ish printed suggestion with:
 - trade_action (BUY/SELL/NO_TRADE)
 - instrument (CE/PE strike chosen)
 - entry_price suggestion
 - stop_loss
 - target_exit
 - expiry_date suggestion
 - notes
"""

import os
import sys
import math
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

# third-party optional
try:
    import pandas as pd
    import numpy as np
except Exception:
    print("Please install pandas and numpy: pip install pandas numpy")
    sys.exit(1)

# Try to import kiteconnect only if user chooses kite mode at runtime
KITE_AVAILABLE = False
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except Exception:
    KITE_AVAILABLE = False

# -----------------------
# Configuration / Helpers
# -----------------------
KEYFILE = "key"

def load_keyfile(path: str = KEYFILE) -> Dict[str, str]:
    cfg = {}
    if not os.path.exists(path):
        # create a template
        with open(path, "w") as f:
            f.write("# MODE=mock or kite\nMODE=mock\n# For kite mode, fill these:\n#KITE_API_KEY=\n#KITE_ACCESS_TOKEN=\n#PREFERRED_EXPIRY_DAYS=7\n")
        print(f"Created template key file at {path}. Edit it and re-run for live mode.")
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            if "=" in line:
                k,v = line.split("=",1)
                cfg[k.strip().upper()] = v.strip()
    return cfg

# -----------------------
# Data fetching (mock + kite)
# -----------------------
def fetch_mock_data_for_backtest() -> pd.DataFrame:
    """Return synthetic BankNifty OHLCV for development/backtest."""
    # generate 1-min-ish synthetic data for last 500 periods
    n = 500
    rng = pd.date_range(end=datetime.now(), periods=n, freq="1T")
    price = 56000 + np.cumsum(np.random.randn(n)) * 8
    df = pd.DataFrame({"datetime": rng, "open": price, "high": price + np.random.rand(n)*6,
                       "low": price - np.random.rand(n)*6, "close": price + np.random.randn(n),
                       "volume": (np.random.rand(n)*100).astype(int)})
    df.set_index("datetime", inplace=True)
    return df

def fetch_live_kite_snapshot(kite, instrument_token: str) -> Dict[str, Any]:
    """
    instrument_token: like 'BANKNIFTY_INDEX' or actual token - this is placeholder.
    Real use: you must map to Kite instrument token for BANKNIFTY and its options.
    """
    # NOTE: user must supply correct instrument tokens / instruments list if using kite.
    snapshot = kite.ltp(instrument_token)
    return snapshot

# -----------------------
# Technical indicators
# -----------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

# -----------------------
# Trend detection
# -----------------------
def detect_market_state(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Returns ('uptrend'|'downtrend'|'sideways', details)
    Uses EMA crossover + ATR-based volatility filter + price vs EMA distance.
    """
    close = df['close']
    ema_short = ema(close, 20)
    ema_long = ema(close, 50)
    atr14 = atr(df, 14).fillna(method='bfill')
    latest = {}
    latest['close'] = float(close.iloc[-1])
    latest['ema20'] = float(ema_short.iloc[-1])
    latest['ema50'] = float(ema_long.iloc[-1])
    latest['atr'] = float(atr14.iloc[-1])
    # Basic logic
    if ema_short.iloc[-1] > ema_long.iloc[-1] and (ema_short.iloc[-1] - ema_long.iloc[-1]) > 0.2 * atr14.iloc[-1]:
        state = 'uptrend'
    elif ema_short.iloc[-1] < ema_long.iloc[-1] and (ema_long.iloc[-1] - ema_short.iloc[-1]) > 0.2 * atr14.iloc[-1]:
        state = 'downtrend'
    else:
        state = 'sideways'
    return state, latest

# -----------------------
# Strike & expiry selection
# -----------------------
def choose_expiry(preferred_days: int = 7) -> Tuple[datetime, str]:
    """
    Very simple expiry chooser:
    - choose the nearest upcoming expiry that's >= preferred_days from today
    NOTE: In reality you'd pull exchange expiry calendar (NSE). Here we approximate weekly expiries (Thurs/Fri).
    """
    today = datetime.now().date()
    # search next 30 days for a Thursday or Friday (typical weekly expiry)
    for d in range(1, 31):
        candidate = today + timedelta(days=d)
        if candidate.weekday() in (3, 4):  # Thu=3, Fri=4
            if (candidate - today).days >= preferred_days:
                return datetime.combine(candidate, datetime.min.time()), candidate.strftime("%Y-%m-%d")
    fallback = today + timedelta(days=preferred_days)
    return datetime.combine(fallback, datetime.min.time()), fallback.strftime("%Y-%m-%d")

def choose_strike(index_price: float, preference: str = "atm", lot_size:int=15) -> Tuple[int, str]:
    """
    preference: 'atm', 'otm1' (one lot OTM), 'itM' etc.
    We'll round strike to nearest 100 for Bank Nifty conventions.
    """
    # Bank Nifty strikes are generally in 50s or 100s; use 100 round for simplicity.
    base = int(round(index_price / 100.0) * 100)
    if preference == "atm":
        strike = base
    elif preference == "otm_call":
        strike = base + 100
    elif preference == "otm_put":
        strike = base - 100
    else:
        strike = base
    return strike, f"{strike}"

# -----------------------
# Position sizing (simple)
# -----------------------
def position_size(capital: float, risk_per_trade_percent: float, stop_loss_points: float, point_value=1.0) -> int:
    """
    Returns number of lots or quantity.
    - capital: total capital in INR
    - risk_per_trade_percent: e.g., 1.0 for 1%
    - stop_loss_points: distance in index points
    - point_value: multiplier: for options, premium-based; for index futures, point value differs
    We'll return quantity in units (not lots) as a rough estimate.
    """
    risk_amount = capital * (risk_per_trade_percent / 100.0)
    if stop_loss_points <= 0:
        return 0
    units = int(risk_amount / (stop_loss_points * point_value))
    return max(units, 0)

# -----------------------
# Trade suggestion logic
# -----------------------
@dataclass
class TradePlan:
    decision: str
    instrument: Optional[str]
    action: Optional[str]
    strike: Optional[int]
    option_type: Optional[str]
    expiry: Optional[str]
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target: Optional[float]
    quantity: Optional[int]
    notes: Optional[str]

def build_trade_plan(df: pd.DataFrame, cfg: Dict[str,str]) -> TradePlan:
    # detect state
    state, details = detect_market_state(df)
    if state == 'sideways':
        return TradePlan(decision="NO_TRADE", instrument=None, action=None, strike=None, option_type=None,
                         expiry=None, entry_price=None, stop_loss=None, target=None, quantity=None,
                         notes="Market sideways — prefer not to trade today.")
    # choose expiry
    pref_days = int(cfg.get("PREFERRED_EXPIRY_DAYS", 7))
    expiry_dt, expiry_str = choose_expiry(pref_days)
    index_price = details['close']
    # choose strike
    # simple: buy ATM option in direction of trend, prefer 1 lot OTM for safety
    if state == 'uptrend':
        opt_type = 'CE'
        strike, _ = choose_strike(index_price, preference="otm_call")
    else:
        opt_type = 'PE'
        strike, _ = choose_strike(index_price, preference="otm_put")
    # entry price logic (very approximate): estimate premium via intrinsic+time value proxy
    # In reality, you'd fetch option LTP. Here we use ATR proxy: premium ~ atr*0.8
    atr14 = atr(df,14).iloc[-1]
    premium_est = max(10.0, float(atr14) * 0.8)  # floor
    # stop loss & target
    # SL = premium * SL_MULT, Target = premium * TG_MULT
    SL_MULT = 1.5
    TG_MULT = 3.0
    stop_loss = round(premium_est * SL_MULT, 2)
    target = round(premium_est * TG_MULT, 2)
    entry_price = round(premium_est,2)
    # Position sizing
    capital = float(cfg.get("CAPITAL", 50000))
    risk_pct = float(cfg.get("RISK_PER_TRADE_PCT", 1.0))
    # For options, point_value is 1 per rupee of premium. Units = number of contracts * lot_size (lot_size placeholder).
    lot_size = int(cfg.get("LOT_SIZE", 15))
    units = position_size(capital, risk_pct, stop_loss, point_value=1.0)
    # Calculate number of lots
    lots = units // lot_size if lot_size>0 else 0
    notes = f"Detected {state}. Chose {opt_type} strike {strike}. Premium estimated by ATR proxy. Replace with real option LTP for live runs."
    return TradePlan(decision="TRADE", instrument=f"BANKNIFTY {strike} {opt_type}", action="BUY",
                     strike=strike, option_type=opt_type, expiry=expiry_str,
                     entry_price=entry_price, stop_loss=stop_loss, target=target,
                     quantity=lots, notes=notes)

# -----------------------
# Simple backtest placeholder
# -----------------------
def simple_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Very simple: simulate entry at next bar and exit when target or SL hit, return winrate.
    This is demonstrative; real backtest requires option price series & realistic slippage.
    """
    # For demo, treat premium ~ ATR*0.8 and simulate.
    atr14 = atr(df,14).fillna(method='bfill')
    premium = (atr14 * 0.8).shift(1).fillna(method='bfill')
    sl_mult = 1.5
    tg_mult = 3.0
    wins = 0
    losses = 0
    trades = 0
    for i in range(1, len(df)-5):
        entry = premium.iloc[i]
        sl = entry * sl_mult
        tg = entry * tg_mult
        # simulate next 5 bars price movement on premium simulated by random walk of underlying
        move = (df['close'].iloc[i+1:i+6].pct_change().fillna(0).cumsum() * df['close'].iloc[i]).abs()
        # rough premium variation
        highs = entry + move.max()/100  # crude
        lows = max(0.1, entry - move.max()/100)
        trades += 1
        if highs >= tg:
            wins += 1
        elif lows <= sl:
            losses += 1
    winrate = (wins / trades) * 100 if trades>0 else 0
    return {"trades": trades, "wins": wins, "losses": losses, "winrate_pct": round(winrate,2)}

# -----------------------
# Main flow
# -----------------------
def main():
    cfg = load_keyfile()
    mode = cfg.get("MODE","mock").lower()
    if mode == "kite":
        if not KITE_AVAILABLE:
            print("kiteconnect library not installed. Install with: pip install kiteconnect")
            return
        kite_api = cfg.get("KITE_API_KEY")
        kite_token = cfg.get("KITE_ACCESS_TOKEN")
        if not kite_api or not kite_token:
            print("Please add KITE_API_KEY and KITE_ACCESS_TOKEN to key file for kite mode.")
            return
        kite = KiteConnect(api_key=kite_api)
        kite.set_access_token(kite_token)
        # NOTE: user must provide instrument token mapping. Here we just error out with instructions.
        print("Live Kite mode enabled -> you must provide instrument tokens and option list integration (not auto-mapped in this script).")
        # For now, fallback to mock feed but notify user
        df = fetch_mock_data_for_backtest()
    else:
        df = fetch_mock_data_for_backtest()
    # Build plan
    plan = build_trade_plan(df, cfg)
    # Backtest summary (quick)
    bt = simple_backtest(df)
    # Print result (JSON-ish)
    out = {
        "timestamp": datetime.now().isoformat(),
        "plan": asdict(plan),
        "backtest_summary": bt
    }
    print(json.dumps(out, indent=2))
    # Human friendly summary
    if plan.decision == "NO_TRADE":
        print("\n>>> SUGGESTION: DO NOT TRADE TODAY — market appears sideways.")
    else:
        print("\n>>> TRADE SUGGESTION:")
        print(f"Instrument: {plan.instrument}")
        print(f"Action: {plan.action}")
        print(f"Expiry: {plan.expiry}")
        print(f"Entry Price (estimate): ₹{plan.entry_price}")
        print(f"Stop Loss (estimate): ₹{plan.stop_loss}")
        print(f"Target (estimate): ₹{plan.target}")
        print(f"Quantity (lots estimated): {plan.quantity} (lot size from key or default 15)")
        print(f"Notes: {plan.notes}")
    print("\nBacktest (very rough) ->", bt)

if __name__ == "__main__":
    main()
