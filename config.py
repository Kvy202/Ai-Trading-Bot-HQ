import os
from dotenv import load_dotenv

load_dotenv()

EXCHANGE_ID      = os.getenv("EXCHANGE_ID", "mexc")
SYMBOL           = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME        = os.getenv("TIMEFRAME", "1h")
LOOKBACK_CANDLES = int(os.getenv("LOOKBACK_CANDLES", "1000"))

RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE", "0.01"))
STOP_ATR_MULT    = float(os.getenv("STOP_ATR_MULT", "2.0"))
TP_R_MULT        = float(os.getenv("TP_R_MULT", "2.0"))
PRED_THRESHOLD   = float(os.getenv("PRED_THRESHOLD", "0.55"))

START_EQUITY     = float(os.getenv("START_EQUITY", "10000"))
FEES_BPS         = float(os.getenv("FEES_BPS", "10"))

LIVE_MODE        = int(os.getenv("LIVE_MODE", "0"))
API_KEY          = os.getenv("API_KEY", "")
API_SECRET       = os.getenv("API_SECRET", "")
API_PASSWORD     = os.getenv("API_PASSWORD", None)

MODEL_PATH       = os.getenv("MODEL_PATH", "models/model.pkl")
LOG_PATH         = os.getenv("LOG_PATH", "logs/trades.csv")
SEED             = int(os.getenv("SEED", "42"))
