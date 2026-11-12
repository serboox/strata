import collections
import threading
import time
from typing import Callable, Optional

SAMPLE_S = 1.0
LOOKAHEAD_S = 3.0
SLOPE_WINDOW_S = 4.0
MIN_SLOPE_SPAN_S = 2.0
RAISE_AFTER_S = 5.0
HOT_BAND_C = 1.0
COOL_BAND_C = 2.0
FAST_DROP_BAND_C = 3.0

DEFAULT_TARGET_C = 77.0
DEFAULT_MIN_MHZ = 1800.0
DEFAULT_MAX_MHZ = 2600.0
DEFAULT_STEP_MHZ = 100.0


def say(message: str) -> None:
    print(message, flush=True)


def next_cap_mhz(cap_mhz: float, lead_c: float, target_c: float, since_change_s: float, lo: float, hi: float, step: float) -> float:
    """The cap for the next second. Above the target the cap falls at once (two steps when the lead is far over); under it the cap
    rises one step, and only after RAISE_AFTER_S without a change, so that the heat of a higher clock shows before the next step."""
    if lead_c >= target_c + FAST_DROP_BAND_C:
        return max(lo, cap_mhz - 2 * step)
    if lead_c >= target_c + HOT_BAND_C:
        return max(lo, cap_mhz - step)
    if lead_c <= target_c - COOL_BAND_C and since_change_s >= RAISE_AFTER_S:
        return min(hi, cap_mhz + step)
    return cap_mhz


class CapController:
    """Moves the CPU frequency cap between lo and hi so that the hottest sensor stays near the target below the pause limit.

    One fixed cap cannot be right for both a short request on a cool package (a higher cap wins) and a long run on a heat-soaked
    one (a lower cap wins); this finds the cap by itself. The engine pause of serve/thermal.py stays as the backstop."""

    def __init__(self, cap, temperatures: Callable[[], dict], target_c: float = DEFAULT_TARGET_C, lo: float = DEFAULT_MIN_MHZ,
                 hi: float = DEFAULT_MAX_MHZ, step: float = DEFAULT_STEP_MHZ, log: Callable[[str], None] = say,
                 sample_s: float = SAMPLE_S, clock: Callable[[], float] = time.monotonic):
        self.cap = cap
        self.temperatures = temperatures
        self.target_c = target_c
        self.lo, self.hi, self.step = lo, hi, step
        self.log = log
        self.sample_s = sample_s
        self.clock = clock
        self.mhz = min(hi, max(lo, cap.max_mhz))
        self.temp_c: Optional[float] = None
        self._changed_at = clock()
        self._histories: dict = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _slope(self, name: str, temp_c: float, now: float) -> float:
        history = self._histories.setdefault(name, collections.deque())
        history.append((now, temp_c))
        while history and now - history[0][0] > SLOPE_WINDOW_S:
            history.popleft()
        span = now - history[0][0]
        return (temp_c - history[0][1]) / span if span >= MIN_SLOPE_SPAN_S else 0.0

    def tick(self) -> float:
        """One decision from the current readings; returns the cap now in force."""
        try:
            readings = {k: float(v) for k, v in self.temperatures().items() if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001 - a sensor hiccup must never stop the server
            readings = {}
        now = self.clock()
        for gone in set(self._histories) - set(readings):
            del self._histories[gone]
        if not readings:
            return self.mhz
        lead = max(value + LOOKAHEAD_S * max(0.0, self._slope(name, value, now)) for name, value in readings.items())
        self.temp_c = max(readings.values())
        wanted = next_cap_mhz(self.mhz, lead, self.target_c, now - self._changed_at, self.lo, self.hi, self.step)
        if wanted != self.mhz:
            self.cap.set_mhz(wanted)
            self.log(f"[strata] cpu governor: {self.temp_c:.0f} C, the cap moves from {self.mhz:.0f} to {wanted:.0f} MHz")
            self.mhz = wanted
            self._changed_at = now
        return self.mhz

    def _loop(self) -> None:
        while not self._stop.wait(self.sample_s):
            self.tick()

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="cpu-cap-controller", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)


def create(cfg: Optional[dict], cap, temperatures: Callable[[], dict], log: Callable[[str], None] = say) -> Optional[CapController]:
    """The controller for the config's "thermal": {"cpu_adaptive": {...}} block, or None when it is absent or there is no cap to move."""
    block = (cfg or {}).get("cpu_adaptive")
    if not block or cap is None:
        return None
    if block is True:
        block = {}
    if not isinstance(block, dict):
        raise ValueError('"thermal.cpu_adaptive" must be true or an object')
    values = {"target_c": DEFAULT_TARGET_C, "min_mhz": DEFAULT_MIN_MHZ, "max_mhz": DEFAULT_MAX_MHZ, "step_mhz": DEFAULT_STEP_MHZ}
    for key in values:
        if key in block:
            value = block[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'"thermal.cpu_adaptive.{key}" must be a number, not {value!r}')
            values[key] = float(value)
    if not values["min_mhz"] <= values["max_mhz"] or values["step_mhz"] <= 0:
        raise ValueError('"thermal.cpu_adaptive" needs min_mhz <= max_mhz and a positive step_mhz')
    return CapController(cap, temperatures, values["target_c"], values["min_mhz"], values["max_mhz"], values["step_mhz"], log=log)
