import atexit
import glob
import os
from typing import Callable, Optional

CPU_ROOT = "/sys/devices/system/cpu"
MIN_MHZ, MAX_MHZ = 800.0, 10000.0
KHZ_PER_MHZ = 1000


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


def reset(root: str = CPU_ROOT) -> list:
    """Puts every limit back to the hardware maximum; returns what could not be written."""
    failures = []
    for path in policy_files(root):
        ceiling = hardware_max_khz(path)
        if ceiling is None:
            continue
        error = _write_text(path, str(ceiling))
        if error is not None:
            failures.append(error)
    return failures


class CpuCap:
    """Keeps the highest frequency every CPU may use at a lower value while the server runs, and puts the old values back at exit.

    The package heats from the arithmetic of the expert kernels, not from reading memory, and the heat per token falls faster
    than the speed when the frequency (and the voltage with it) goes down, so a lower cap lets the model run without pauses.
    A limit that somebody else changed while the cap was held is left as that program set it."""

    def __init__(self, max_mhz: float, log: Callable[[str], None] = say, root: str = CPU_ROOT):
        self.max_mhz = max_mhz
        self.log = log
        self.root = root
        self._held: dict = {}
        self._registered = False

    def start(self) -> None:
        if not self._registered:
            atexit.register(self.stop)
            self._registered = True
        self.apply()

    def apply(self) -> list:
        """Writes the cap; returns the failures. The old value is kept before the write, so an interrupt cannot lose it."""
        files = policy_files(self.root)
        if not files:
            self.log("[strata] cpu cap: this machine has no cpufreq limit to set, the cap is not applied")
            return []
        failures = []
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
            self._held[path] = (original, str(target))
            error = _write_text(path, str(target))
            if error is not None:
                del self._held[path]
                failures.append(error)
        if failures:
            self.log(f"[strata] cpu cap: {len(failures)} of {len(files)} limits could not be written ({failures[0]}); "
                     f"see the frequency cap in docs/TUNING.md for the tmpfiles rule that makes them writable")
        if self._held:
            self.log(f"[strata] cpu cap: the highest CPU frequency is held at {self.max_mhz:.0f} MHz on {len(self._held)} policies")
        return failures

    def stop(self) -> None:
        for path, (original, written) in list(self._held.items()):
            if _read_text(path) != written:
                del self._held[path]
                continue
            error = _write_text(path, original)
            if error is None:
                del self._held[path]
        if self._held:
            self.log(f"[strata] cpu cap: {len(self._held)} limits could not be put back; run tools/cpucap reset")


def create(cfg: Optional[dict], log: Callable[[str], None] = say, root: str = CPU_ROOT) -> Optional[CpuCap]:
    """The cap for the config's "thermal" block (key "cpu_max_mhz"), or None when it is absent or 0."""
    value = (cfg or {}).get("cpu_max_mhz")
    if value is None or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not MIN_MHZ <= value <= MAX_MHZ:
        raise ValueError(f'"thermal.cpu_max_mhz" must be 0 (off) or a number from {MIN_MHZ:.0f} to {MAX_MHZ:.0f}, not {value!r}')
    return CpuCap(float(value), log=log, root=root)
