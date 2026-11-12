import atexit
import fcntl
import glob
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

CPU_ROOT = "/sys/devices/system/cpu"
MIN_MHZ, MAX_MHZ = 800.0, 10000.0
KHZ_PER_MHZ = 1000
STATE_ENV = "STRATA_CPUCAP_STATE"


def say(message: str) -> None:
    print(message, flush=True)


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="ascii") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _write_text(path: str, value: str) -> Optional[str]:
    """None when the value was written, else why not."""
    try:
        with open(path, "w", encoding="ascii") as fh:
            fh.write(value)
        return None
    except OSError as error:
        return str(error)


def limit_files(root: str = CPU_ROOT) -> list:
    return sorted(glob.glob(os.path.join(root, "cpu[0-9]*", "cpufreq", "scaling_max_freq")))


def policy_files(root: str = CPU_ROOT) -> list:
    """One limit file per cpufreq policy: CPUs that share a policy point at the same directory and must be handled once."""
    seen, files = set(), []
    for path in limit_files(root):
        policy = os.path.realpath(os.path.dirname(path))
        if policy not in seen:
            seen.add(policy)
            files.append(path)
    return files


def hardware_max_khz(limit_file: str) -> Optional[int]:
    value = _read_text(os.path.join(os.path.dirname(limit_file), "cpuinfo_max_freq"))
    return int(value) if value and value.isdigit() else None


def describe(root: str = CPU_ROOT) -> str:
    files = limit_files(root)
    if not files:
        return "no cpufreq limit on this machine"
    limits = sorted({_read_text(path) for path in files} - {None})
    ceiling = hardware_max_khz(files[0])
    shown = ", ".join(f"{int(value) // KHZ_PER_MHZ} MHz" for value in limits if value.isdigit())
    return f"highest CPU frequency {shown} on {len(files)} CPUs" + (f" (hardware maximum {ceiling // KHZ_PER_MHZ} MHz)" if ceiling else "")


def state_path() -> str:
    """Where a running server records the limits it holds, so that a guard can put them back if the server is killed."""
    override = os.environ.get(STATE_ENV)
    if override:
        return override
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and os.path.isdir(runtime):
        return os.path.join(runtime, "strata-cpucap.json")
    return os.path.join(tempfile.gettempdir(), f"strata-cpucap-{os.getuid()}.json")


@contextmanager
def _locked(record_path: str) -> Iterator[None]:
    """One process at a time works on the record: a guard, a starting server and a stopping one must not interleave."""
    with open(f"{record_path}.lock", "a", encoding="ascii") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _process_start(pid: int) -> Optional[str]:
    """The start time of a process in clock ticks; it tells a reused pid from the process that held the cap."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _is_limit_pair(values) -> bool:
    return isinstance(values, list) and len(values) == 2 and all(isinstance(value, str) for value in values)


def _load_state(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not (isinstance(data, dict) and isinstance(data.get("pid"), int) and isinstance(data.get("start"), str)):
        return None
    held = data.get("held")
    if isinstance(held, dict) and all(_is_limit_pair(values) for values in held.values()):
        return data
    return None


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _recover_locked(path: str, log: Callable[[str], None]) -> list:
    if not os.path.exists(path):
        return []
    data = _load_state(path)
    if data is None:
        _remove(path)
        problem = "the record of the held limits was damaged and is removed; check the limits with tools/cpucap status"
        log(f"[strata] cpu cap: {problem}")
        return [problem]
    if _process_start(data["pid"]) == data["start"]:
        return []
    failures, restored = [], 0
    for limit_file, (original, written) in data["held"].items():
        if not os.path.exists(limit_file):
            continue
        current = _read_text(limit_file)
        if current is None:
            failures.append(f"{limit_file} cannot be read")
        elif current == written:
            error = _write_text(limit_file, original)
            if error is None:
                restored += 1
            else:
                failures.append(error)
    if failures:
        log(f"[strata] cpu cap: {len(failures)} limits could not be put back after process {data['pid']} ended ({failures[0]})")
        return failures
    _remove(f"{path}.tmp")
    _remove(path)
    if restored:
        log(f"[strata] cpu cap: {restored} limits put back, the process {data['pid']} that held them ended without doing it")
    return []


def recover_stale(state: Optional[str] = None, log: Callable[[str], None] = say) -> list:
    """Puts back the limits that a server held when it ended without doing it (killed, crashed); returns the failures.

    A limit that somebody else changed since is left alone, and so is a record whose server is still running."""
    path = state or state_path()
    if not os.path.exists(path):
        return []
    try:
        with _locked(path):
            return _recover_locked(path, log)
    except OSError as error:
        log(f"[strata] cpu cap: the record of the held limits could not be locked ({error})")
        return [str(error)]


def _reset_unlocked(root: str, state: str) -> list:
    failures = []
    for path in policy_files(root):
        ceiling = hardware_max_khz(path)
        if ceiling is None:
            continue
        error = _write_text(path, str(ceiling))
        if error is not None:
            failures.append(error)
    if not failures:
        _remove(state)
    return failures


def reset(root: str = CPU_ROOT, state: Optional[str] = None) -> list:
    """Puts every limit back to the hardware maximum; returns what could not be written."""
    path = state or state_path()
    try:
        with _locked(path):
            return _reset_unlocked(root, path)
    except OSError:
        return _reset_unlocked(root, path)


class CpuCap:
    """Keeps the highest frequency every CPU may use at a lower value while the server runs, and puts the old values back at exit.

    A lower frequency (and the voltage with it) lowers the heat per token faster than the speed, so the temperature limit pauses
    the engine less often.
    A limit that somebody else changed while the cap was held is left as that program set it. Before the first limit is written the
    server saves what it is going to hold in a state file, and it holds nothing when that fails; a guard (`tools/cpucap guard`)
    puts the limits back from the file if the server is killed and cannot do it. A second server that starts while the first one
    runs leaves the cap alone, so there is always one owner of the record."""

    def __init__(self, max_mhz: float, log: Callable[[str], None] = say, root: str = CPU_ROOT, state: Optional[str] = None):
        self.max_mhz = max_mhz
        self.log = log
        self.root = root
        self.state = state
        self._held: dict = {}
        self._registered = False

    def start(self) -> None:
        if not self._registered:
            atexit.register(self.stop)
            self._registered = True
        record = self.state or state_path()
        try:
            with _locked(record):
                if _recover_locked(record, self.log):
                    self.log("[strata] cpu cap: not applied, the limits of an earlier server are not put back yet")
                    return
                holder = _load_state(record)
                if holder is not None and holder["pid"] != os.getpid():
                    self.log(f"[strata] cpu cap: not applied, the server with process id {holder['pid']} holds the cap")
                    return
                self.apply(record=record)
        except OSError as error:
            self.log(f"[strata] cpu cap: the record for the guard could not be locked ({error}); the cap is not applied")

    def _write_record(self, path: str) -> Optional[str]:
        """None when the record is in place or not needed, else why not. The record must be locked by the caller."""
        if not self._held:
            data = _load_state(path)
            if data is not None and data["pid"] == os.getpid():
                _remove(path)
            return None
        start = _process_start(os.getpid())
        if start is None:
            return "the start time of this process cannot be read"
        record = {"pid": os.getpid(), "start": start, "held": {limit_file: list(values) for limit_file, values in self._held.items()}}
        temporary = f"{path}.tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            os.replace(temporary, path)
        except OSError as error:
            _remove(temporary)
            return str(error)
        return None

    def apply(self, record: Optional[str] = None) -> list:
        """Writes the cap; returns the failures. With a record path, the old values are saved there before the first write."""
        files = policy_files(self.root)
        if not files:
            self.log("[strata] cpu cap: this machine has no cpufreq limit to set, the cap is not applied")
            return []
        plan = {}
        for path in files:
            if path in self._held:
                continue
            original = _read_text(path)
            if original is None:
                continue
            target = int(self.max_mhz * KHZ_PER_MHZ)
            ceiling = hardware_max_khz(path)
            if ceiling is not None:
                target = min(target, ceiling)
            plan[path] = (original, str(target))
        self._held.update(plan)
        if record is not None:
            error = self._write_record(record)
            if error is not None:
                for path in plan:
                    del self._held[path]
                self.log(f"[strata] cpu cap: the record for the guard could not be saved ({error}); the cap is not applied")
                return [error]
        failures = []
        for path, (_, target) in plan.items():
            error = _write_text(path, target)
            if error is not None:
                del self._held[path]
                failures.append(error)
        if failures:
            self.log(f"[strata] cpu cap: {len(failures)} of {len(files)} limits could not be written ({failures[0]}); "
                     f"see the frequency cap in docs/TUNING.md for the tmpfiles rule that makes them writable")
            if record is not None:
                self._write_record(record)
        if self._held:
            self.log(f"[strata] cpu cap: the highest CPU frequency is held at {self.max_mhz:.0f} MHz on {len(self._held)} policies")
        return failures

    def set_mhz(self, mhz: float) -> bool:
        """Moves the held cap to another value. A limit that somebody else changed since the cap wrote it is left alone.
        True when at least one limit was written. The record for the guard is rewritten with the new value."""
        self.max_mhz = mhz
        if not self._held:
            return False
        record = self.state or state_path()
        changed = False
        try:
            with _locked(record):
                for path, (original, written) in list(self._held.items()):
                    target = int(mhz * KHZ_PER_MHZ)
                    ceiling = hardware_max_khz(path)
                    if ceiling is not None:
                        target = min(target, ceiling)
                    if str(target) == written:
                        continue
                    current = _read_text(path)
                    if not os.path.exists(path) or (current is not None and current != written):
                        del self._held[path]
                        continue
                    if _write_text(path, str(target)) is None:
                        self._held[path] = (original, str(target))
                        changed = True
                self._write_record(record)
        except OSError as error:
            self.log(f"[strata] cpu cap: the cap could not be moved ({error})")
        return changed

    def _put_back(self) -> None:
        for path, (original, written) in list(self._held.items()):
            current = _read_text(path)
            if not os.path.exists(path) or (current is not None and current != written):
                del self._held[path]
                continue
            error = _write_text(path, original)
            if error is None:
                del self._held[path]
        if self._held:
            self.log(f"[strata] cpu cap: {len(self._held)} limits could not be put back; run tools/cpucap reset")

    def stop(self) -> None:
        if not self._registered:
            self._put_back()
            return
        record = self.state or state_path()
        try:
            with _locked(record):
                self._put_back()
                error = self._write_record(record)
        except OSError as error:
            self.log(f"[strata] cpu cap: the record for the guard could not be updated ({error})")
            self._put_back()
            return
        if error is not None:
            self.log(f"[strata] cpu cap: the record for the guard could not be updated ({error})")


def create(cfg: Optional[dict], log: Callable[[str], None] = say, root: str = CPU_ROOT) -> Optional[CpuCap]:
    """The cap for the config's "thermal" block (key "cpu_max_mhz"), or None when it is absent or 0."""
    value = (cfg or {}).get("cpu_max_mhz")
    if value is None or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not MIN_MHZ <= value <= MAX_MHZ:
        raise ValueError(f'"thermal.cpu_max_mhz" must be 0 (off) or a number from {MIN_MHZ:.0f} to {MAX_MHZ:.0f}, not {value!r}')
    return CpuCap(float(value), log=log, root=root)
