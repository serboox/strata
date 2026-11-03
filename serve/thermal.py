from __future__ import annotations

import atexit
import collections
import glob
import os
import signal
import threading
import time
from typing import Callable, Optional

CPU_SENSOR_NAMES = ("coretemp", "k10temp", "zenpower")
CPU_PACKAGE_LABELS = ("Package", "Tctl", "Tdie")
PERIOD_S = 0.5
SAMPLE_S = 1.0
RESUME_MARGIN_C = 3.0
MAX_DUTY = 0.85
ACTIVE_DUTY = 0.02
MIN_LIMIT_C, MAX_LIMIT_C = 40.0, 110.0
LOOKAHEAD_S = 3.0
SLOPE_WINDOW_S = 4.0
MIN_SLOPE_SPAN_S = 2.0


def say(message: str) -> None:
    print(message, flush=True)


def _read_number(path: str) -> Optional[float]:
    try:
        with open(path, encoding="ascii") as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="ascii", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def cpu_package_temp_c(hwmon_root: str = "/sys/class/hwmon", thermal_root: str = "/sys/class/thermal") -> Optional[float]:
    """The hottest CPU package reading in degrees Celsius, or None when the machine exposes none."""
    readings = []
    for hwmon in sorted(glob.glob(os.path.join(hwmon_root, "hwmon*"))):
        if _read_text(os.path.join(hwmon, "name")) not in CPU_SENSOR_NAMES:
            continue
        for input_path in sorted(glob.glob(os.path.join(hwmon, "temp*_input"))):
            label = _read_text(input_path.replace("_input", "_label"))
            if label.startswith(CPU_PACKAGE_LABELS) or (not label and input_path.endswith("temp1_input")):
                value = _read_number(input_path)
                if value is not None:
                    readings.append(value / 1000.0)
    if not readings:
        for zone in sorted(glob.glob(os.path.join(thermal_root, "thermal_zone*"))):
            if _read_text(os.path.join(zone, "type")) == "x86_pkg_temp":
                value = _read_number(os.path.join(zone, "temp"))
                if value is not None:
                    readings.append(value / 1000.0)
    return max(readings) if readings else None


def next_duty(temp_c: float, duty: float, limit_c: float, margin_c: float = RESUME_MARGIN_C, max_duty: float = MAX_DUTY,
              slope_c_s: float = 0.0) -> float:
    """The share of each period the engine is paused for, given the hottest reading now. A rising temperature counts as the
    temperature it will have in LOOKAHEAD_S seconds, so the pause starts before the limit is crossed: the package heats by a
    few degrees a second under full load and keeps rising for a moment after the pause begins."""
    ahead_c = temp_c + LOOKAHEAD_S * max(0.0, slope_c_s)
    if ahead_c >= limit_c:
        return min(max_duty, duty + 0.10 + 0.04 * (ahead_c - limit_c))
    if ahead_c <= limit_c - margin_c:
        return max(0.0, duty - 0.05)
    return duty


class Governor:
    """Holds the machine at or under a temperature by pausing the engine process for part of every period.

    A paused process draws no CPU and starts no GPU work, so both cool. The pause never outlives a period, and the
    engine is released on stop, at exit and (through its parent-death signal) if this server is killed."""

    def __init__(self, limit_c: float, temperatures: Callable[[], dict], get_pid: Callable[[], Optional[int]],
                 log: Callable[[str], None] = say, period_s: float = PERIOD_S, sample_s: float = SAMPLE_S,
                 kill: Callable[[int, int], None] = os.kill, clock: Callable[[], float] = time.monotonic):
        self.limit_c = limit_c
        self.temperatures = temperatures
        self.get_pid = get_pid
        self.log = log
        self.period_s = period_s
        self.sample_s = sample_s
        self.kill = kill
        self.clock = clock
        self._histories: dict = {}
        self.duty = 0.0
        self.temp_c: Optional[float] = None
        self.hottest: Optional[str] = None
        self.throttled_s = 0.0
        self._announced = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="thermal-governor", daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._signal(signal.SIGCONT)
        self.duty = 0.0

    def status(self) -> dict:
        return {"limit_c": self.limit_c, "temp_c": self.temp_c, "hottest": self.hottest, "duty": round(self.duty, 3),
                "throttled_s": round(self.throttled_s, 1)}

    def _send(self, pid: Optional[int], sig: int) -> None:
        if not pid:
            return
        try:
            self.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _signal(self, sig: int) -> None:
        self._send(self.get_pid(), sig)

    def _slope(self, source: str, temp_c: float) -> float:
        """Degrees per second of one sensor over the last few seconds of its readings; 0 until they span long enough. Every sensor
        keeps its own history, so a hot steady sensor cannot hide a cooler one that is heating fast."""
        now = self.clock()
        history = self._histories.setdefault(source, collections.deque())
        if history:
            gap = now - history[-1][0]
            if gap == 0.0:
                history.pop()
            elif gap < 0.0:
                history.clear()
        # A slow sample is not a reason to forget the history: a slope over a longer span only pauses sooner, while an empty
        # history blinds the look-ahead for the next two seconds.
        history.append((now, temp_c))
        while history and now - history[0][0] > SLOPE_WINDOW_S:
            history.popleft()
        oldest_time, oldest_temp = history[0]
        span = now - oldest_time
        return (temp_c - oldest_temp) / span if span >= MIN_SLOPE_SPAN_S else 0.0

    def _sample(self) -> None:
        try:
            readings = {k: v for k, v in self.temperatures().items() if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001 - a sensor hiccup must never stop the server
            readings = {}
        for gone in set(self._histories) - set(readings):
            del self._histories[gone]   # a sensor that returns starts a fresh history instead of a false rise
        if not readings:
            self.temp_c, self.hottest = None, None
            self.duty = max(0.0, self.duty - 0.05)
            return
        self.hottest = max(readings, key=readings.get)
        self.temp_c = float(readings[self.hottest])
        slopes = {name: self._slope(name, float(value)) for name, value in readings.items()}
        lead = max(readings, key=lambda name: float(readings[name]) + LOOKAHEAD_S * max(0.0, slopes[name]))
        previous = self.duty
        self.duty = next_duty(float(readings[lead]), self.duty, self.limit_c, slope_c_s=slopes[lead])
        if previous <= ACTIVE_DUTY < self.duty:
            self.log(f"[strata] thermal: {self.hottest} {self.temp_c:.0f} C reached the {self.limit_c:.0f} C limit, "
                     f"the engine now pauses {self.duty:.0%} of the time")
            self._announced = self.duty
        elif previous > ACTIVE_DUTY >= self.duty:
            self.log(f"[strata] thermal: {self.hottest} {self.temp_c:.0f} C, the limit holds, the engine runs freely again")
            self._announced = 0.0
        elif abs(self.duty - self._announced) >= 0.2 and self.duty > ACTIVE_DUTY:
            self.log(f"[strata] thermal: {self.hottest} {self.temp_c:.0f} C, the engine now pauses {self.duty:.0%} of the time")
            self._announced = self.duty

    def _loop(self) -> None:
        next_sample = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_sample:
                self._sample()
                next_sample = now + self.sample_s
            if self.duty <= ACTIVE_DUTY:
                self._stop.wait(self.period_s)
                continue
            pause = self.duty * self.period_s
            # The resume goes to the process that was stopped, even if the engine was replaced in between.
            pid = self.get_pid()
            self._send(pid, signal.SIGSTOP)
            try:
                self._stop.wait(pause)
            finally:
                self._send(pid, signal.SIGCONT)
            self.throttled_s += pause
            self._stop.wait(self.period_s - pause)


def create(cfg: Optional[dict], temperatures: Callable[[], dict], get_pid: Callable[[], Optional[int]],
           log: Callable[[str], None] = say) -> Optional[Governor]:
    """The governor for the config's "thermal" block, or None when it is off or the platform cannot pause a process."""
    if not cfg:
        return None
    limit = cfg.get("max_temp_c")
    if limit is None or limit == 0:
        return None
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not MIN_LIMIT_C <= limit <= MAX_LIMIT_C:
        raise ValueError(f'"thermal.max_temp_c" must be 0 (off) or a number from {MIN_LIMIT_C:.0f} to {MAX_LIMIT_C:.0f}, not {limit!r}')
    if not hasattr(signal, "SIGSTOP"):
        log("[strata] thermal: this platform cannot pause the engine, the temperature limit is not enforced")
        return None
    return Governor(float(limit), temperatures, get_pid, log=log)
