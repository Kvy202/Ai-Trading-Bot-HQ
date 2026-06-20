import os, time, threading, requests
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Any

# -------- TraderState: simple thread-safe key/value store --------
class TraderState:
    """
    Flexible state holder with a small API:
      - snapshot() -> dict copy of current state
      - update(**kw) -> update keys atomically
      - get(key, default)
    """
    def __init__(self, **kwargs):
        self._lock = threading.RLock()
        self._data = {
            "paused": False,
            "running": False,
            "pred_threshold": 0.60,
            "max_concurrent": 1,
            "scan_topn": 4,
            "timeframe": "5m",
        }
        self._data.update(kwargs)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def update(self, **kw):
        with self._lock:
            self._data.update(kw)

    def get(self, k, default=None):
        with self._lock:
            return self._data.get(k, default)

# -------- TelegramBot: minimal long-polling bot (no extra libs) --------
class TelegramBot:
    """
    Lightweight Telegram bot using getUpdates long polling (requests only).

    Commands:
      /start, /help, /status, /pause, /resume, /stop, /restart
      /th <0.55>           -> set prediction threshold
      /maxc <N>            -> set MAX_CONCURRENT
      /scan <N>            -> set SCAN_TOPN
      /positions           -> show current positions
      /close <SYMBOL|ALL>  -> close specific position or all
      /report              -> send previous-day PnL report
      /model_status        -> show DL model readiness + latest inferences
    """
    def __init__(
        self,
        token: str,
        state: TraderState,
        *,
        callbacks: Optional[Dict[str, Callable[..., Any]]] = None,
        storage_dir: str = "logs",
        daily_report_hhmm: str = "00:05",
        enable_poll: bool = True,
    ):
        self.token = (token or "").strip()
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
        self.base = f"https://api.telegram.org/bot{self.token}"

        self.state = state
        self.callbacks = callbacks or {}

        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        # chat id memory
        self.chat_id_file = os.path.join(self.storage_dir, "telegram_chat_id.txt")
        self.chat_id: Optional[int] = None
        cid_env = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if cid_env.startswith("-") or cid_env.isdigit():
            try:
                self.chat_id = int(cid_env)
            except Exception:
                self.chat_id = None
        else:
            # try stored file
            try:
                if os.path.exists(self.chat_id_file):
                    with open(self.chat_id_file, "r", encoding="utf-8") as f:
                        cid = f.read().strip()
                        if cid.startswith("-") or cid.isdigit():
                            self.chat_id = int(cid)
            except Exception:
                pass

        # lifecycle
        self.enable_poll = enable_poll
        self._stop = threading.Event()
        self._offset = 0
        self.daily_report_hhmm = daily_report_hhmm or "00:05"
        self._poll_th: Optional[threading.Thread] = None
        self._daily_th: Optional[threading.Thread] = None

    # ---------- lifecycle ----------
    def start(self):
        if self.enable_poll:
            self._poll_th = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_th.start()
        self._daily_th = threading.Thread(target=self._daily_loop, daemon=True)
        self._daily_th.start()

    # alias for familiarity
    def start_polling(self):
        self.start()

    def stop(self):
        self._stop.set()

    # ---------- messaging ----------
    def send(self, text: str, chat_id: Optional[int] = None) -> bool:
        cid = chat_id or self.chat_id
        if not cid:
            return False
        try:
            requests.post(self.base + "/sendMessage", json={"chat_id": cid, "text": text}, timeout=15)
            return True
        except Exception:
            return False

    def safe_send(self, text: str):
        try:
            self.send(text)
        except Exception:
            pass

    def _save_chat_id(self, cid: int):
        self.chat_id = cid
        try:
            with open(self.chat_id_file, "w", encoding="utf-8") as f:
                f.write(str(cid))
        except Exception:
            pass

    # ---------- command handling ----------
    def _handle_cmd(self, cid: int, txt: str):
        t = txt.strip()
        if not t.startswith("/"):
            return
        parts = t.split()
        cmd = parts[0].lower()
        arg = parts[1:] if len(parts) > 1 else []
        cb = lambda name: self.callbacks.get(name)

        if cmd == "/start":
            self._save_chat_id(cid)
            self.send(self._help_text(), cid)
            st = self.state.snapshot()
            self.send(f"Connected. paused={st.get('paused')} running={st.get('running')}", cid)

        elif cmd == "/help":
            self.send(self._help_text(), cid)

        elif cmd == "/status":
            if cb("on_status"): self.send(cb("on_status")(), cid)

        elif cmd == "/pause":
            if cb("on_pause"): self.send(cb("on_pause")(), cid)

        elif cmd == "/resume":
            if cb("on_resume"): self.send(cb("on_resume")(), cid)

        elif cmd == "/stop":
            if cb("on_stop"): self.send(cb("on_stop")(), cid)

        elif cmd == "/restart":
            if cb("on_restart"): self.send(cb("on_restart")(), cid)

        elif cmd == "/report":
            if cb("on_report"): self.send(cb("on_report")(), cid)

        elif cmd == "/positions":
            if cb("on_positions"): self.send(cb("on_positions")(), cid)

        elif cmd == "/th":
            if not arg:
                self.send("Usage: /th 0.60", cid); return
            try:
                v = float(arg[0])
                if cb("on_set_th"): self.send(cb("on_set_th")(v), cid)
            except Exception:
                self.send("Invalid number. Example: /th 0.58", cid)

        elif cmd == "/maxc":
            if not arg:
                self.send("Usage: /maxc 2", cid); return
            try:
                v = int(arg[0])
                if cb("on_set_maxc"): self.send(cb("on_set_maxc")(v), cid)
            except Exception:
                self.send("Invalid integer. Example: /maxc 2", cid)

        elif cmd == "/scan":
            if not arg:
                self.send("Usage: /scan 4", cid); return
            try:
                v = int(arg[0])
                if cb("on_set_scan"): self.send(cb("on_set_scan")(v), cid)
            except Exception:
                self.send("Invalid integer. Example: /scan 4", cid)

        elif cmd == "/close":
            if not arg:
                self.send("Usage: /close ALL  OR  /close DOGE/USDT:USDT", cid); return
            what = " ".join(arg).strip()
            if what.upper() == "ALL":
                if cb("on_close_all"): self.send(cb("on_close_all")(), cid)
            else:
                if cb("on_close_symbol"): self.send(cb("on_close_symbol")(what), cid)

        # ---------- NEW ----------
        elif cmd == "/model_status":
            if cb("on_model_status"):
                self.send(cb("on_model_status")(), cid)
            else:
                self.send("DL pocket not wired. Set USE_DL_SIGNALS=1 and provide artifacts.", cid)

        else:
            self.send("Unknown command. /help", cid)

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                r = requests.get(self.base + "/getUpdates",
                                 params={"timeout": 50, "offset": self._offset + 1},
                                 timeout=60)
                j = r.json()
                if not j.get("ok"):
                    time.sleep(2); continue
                for upd in j.get("result", []):
                    self._offset = int(upd["update_id"])
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg: continue
                    cid = msg["chat"]["id"]
                    txt = msg.get("text") or ""
                    if txt:
                        self._handle_cmd(cid, txt)
            except Exception:
                time.sleep(2)

    def _daily_loop(self):
        # tick every 30s and see if we crossed the scheduled wall time
        hh, mm = (self.daily_report_hhmm or "00:05").split(":")
        hh = int(hh); mm = int(mm)
        last_sent_date = None
        while not self._stop.is_set():
            now = datetime.now(timezone.utc).astimezone()
            target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if now >= target and (last_sent_date is None or last_sent_date != now.date()):
                cb = self.callbacks.get("on_report")
                if cb:
                    try:
                        text = cb()
                        if text and self.chat_id:
                            self.send(text)
                    except Exception:
                        pass
                last_sent_date = now.date()
            time.sleep(30)

    @staticmethod
    def _help_text() -> str:
        return (
            "Commands:\n"
            "/status – bot status\n"
            "/pause | /resume – pause/resume entries\n"
            "/stop | /restart – stop or restart the trader\n"
            "/th <0.55> – set prediction threshold\n"
            "/maxc <N> – set max concurrent positions\n"
            "/scan <N> – set scan top-N\n"
            "/positions – show open positions\n"
            "/close <SYMBOL|ALL> – close a position or all\n"
            "/report – send previous-day PnL report\n"
            "/model_status – DL model readiness + last scores\n"
            "/help – this message"
        )
