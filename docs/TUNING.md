# Tuning notes

Measured on one machine: i7-11370H (4 cores, 8 threads, 54 GB/s peak memory read), RTX 3050 Ti Laptop (4 GB), 31 GB RAM,
NVMe at 3.5 GB/s, Qwen3.6-35B-A3B UD-Q4_K_XL (22.9 GB). Numbers are tokens per second.

## Memory budget against speed

Decoding reads about 630 MB of routed experts per token. What is not in the page cache comes from the SSD, so the budget
(`--ram`, a cgroup limit that counts the page cache) is the main lever.

Greedy decoding with MTP, real code text, 900 tokens per request, started cold and warmed by two requests, each request
started with the CPU package at 55 C or less:

| `--ram` | profile | first build | now |
| --- | --- | --- | --- |
| 8G | `lean` | 9.4 | 10.3 |
| 12G | `balanced` | 12.7 | 17.3 |
| 16G | `fast` | 17.1 | 25.4 |
| 22G | `max` | 21.9 | 33.1 |
| none | `auto` (default) | 32.1 / 32.0 | 36.2 / 36.9 / 39.1 (after the AVX-512 kernels and the single barrier) |

The "first build" column was measured with 300-token requests and no cooling, so it is slightly optimistic against the
"now" column. With the whole model in the page cache the engine alone sustains 33-37 tok/s over 900 tokens.

The CPU package reaches 100 C about ten seconds into a generation and the clocks then move between 3.2 and 4.3 GHz: a
short request can show 37 tok/s while the same request after a minute of load shows 25-28. Compare runs only from the
same thermal state.

## Keeping the model in memory

Two different things can take the model out of memory. The swap takes memory of a process (the engine's buffers, the Python
server), and the server's scope has `MemorySwapMax=0` when a budget is set, so it does not swap. The model file is another
case: it sits in the page cache, which the kernel drops when another program needs the memory and reads back from the SSD on
the next access. That is what makes the speed fall under pressure (the table above), and the `madvise(WILLNEED)` in the engine
only asks for a read-ahead, it pins nothing.

`--protect-ram` asks the kernel not to drop it: the server's scope gets `MemoryLow` of the model size plus 2 GiB, capped by the
budget. The kernel then takes memory from every unprotected program first and touches the model only when nothing else is
left. It is a soft guarantee, so it cannot cause an out-of-memory kill of another program. It applies only to the profiles
that read through the page cache (`auto` and `max`); the `direct` profiles never fill it.

The `MemoryLow` of a scope is honoured only up to the lowest `MemoryLow` of every level above it: `user.slice`,
`user-1000.slice`, `user@1000.service` and the `app.slice` of the user manager. Without a value on each of them the setting
does nothing. On this machine each is 24G, set by a drop-in `strata-memory-low.conf` in `/etc/systemd/system/user.slice.d`,
`user-.slice.d`, `user@.service.d` and in `/etc/systemd/user/app.slice.d`. The launcher reads these levels (found from the user
id) and prints a warning when one allows less than the server asks. That is a necessary condition, not a guarantee: the kernel
shares the value of a level in proportion among every scope below it that asks for protection. By hand:

    for d in user.slice user.slice/user-1000.slice user.slice/user-1000.slice/user@1000.service \
             user.slice/user-1000.slice/user@1000.service/app.slice; do echo "$d $(cat /sys/fs/cgroup/$d/memory.low)"; done

`--prewarm` reads the model file once before the server starts, in the same scope, so the pages are charged to it and
protected by it. A cached page stays charged to the control group that read it first (the scope of an earlier run, or a shell),
so the file is dropped from the cache and read again; a 1 GiB test file that another group had cached ended up as 1024 MiB in
the scope's own page cache, and the read took under a second on the NVMe disk. The re-read costs the time of reading the file
from the SSD even when it was cached. A page that another process has mapped stays with its owner, so the helper reports how
much of the file ended up counted in the scope and warns when it is less than 90 %. A scope is created for the prewarm even
when no other property asks for one; a probe checks that a user manager is reachable, and when it is not the launcher warns,
leaves the protection off and runs the prewarm in the current control group. It is skipped when the profile reads with O_DIRECT, when the model
does not fit the budget, and when less memory is available than the model needs, because reading it would push other
programs out. Both are on by default and switch off with `--no-protect-ram` and `--no-prewarm`; with both switched off a
start is the same as before, and the `direct` profiles (`lean`, `balanced`, `fast`) are the same either way. The prewarm
makes the server ready later by the time the file takes to read.

Checked: the values the kernel reports after the drop-ins, the scope's own `memory.stat` after a prewarm (a 1 GiB test file
shows up as 1024 MiB of the scope's page cache), the launcher's command and warnings with `--dry-run`, and that a start with both
flags switched off prints the same command as before. Not measured: how the model behaves under real memory pressure with
`--protect-ram`.

## What each setting does

| Setting | Value | Effect |
| --- | --- | --- |
| `--ram` / `--profile` | none (`auto`) | page-cache budget, see above; without a limit the kernel decides and gives memory back on demand |
| `--context` | 65536 | window; the K/V cache is 0.73 GiB in q8, 1.38 GiB in f16 |
| `--att-scores` | 4 | millions of floats of the prompt attention's score buffer; 8 doubles the tile of keys and the buffer (12 MiB more), no speed difference measured |
| `--chunk` | 512 | prompt tokens per pass; each pass streams every expert from the SSD once, so prompt speed grows with the chunk: 44 tok/s at 256, 65 at 384, 84 at 512, 123 at 768 (3099-token prompt, 9G) |
| `--kv` | q8 | 34 bytes per 32 values; f16 is available and token-identical on the parity prompts |
| `--workers` | 3 | CPU expert threads besides the main one; 4 is slower, 7 is half the speed |
| `--mtp` | 2 | drafts per step, about 90% accepted; 3 and 4 were not faster |
| `--sampling` | greedy | drafting runs only for greedy requests, `card` turns it off |
| `--max-temp` | 80 | while the CPU package or the GPU is at this many degrees or hotter, the engine is paused for a share of every half second, a bigger share the hotter it is; 0 turns the limit off |
| `--turbo` | off | the CPU turbo boost at start: `off`, `on` or `keep`; `tools/turbo on|off|status` switches it at any time |
| `--priority` | background | the server and the engine yield the CPU and the disk to every other program (see "Staying in the background"); `normal` leaves the priorities alone |
| `--protect-ram` | on | ask the kernel to keep the cached model in memory (`MemoryLow` on the server's scope); soft, and only fully honoured when every level above allows it (see "Keeping the model in memory"); `--no-protect-ram` turns it off |
| `--prewarm` | on | read the model file into the page cache before the server starts; skipped when the memory is not free for it; `--no-prewarm` turns it off |
| `--prefetch` | on | read-ahead of an expert's ranges, issued only when the range is not cached |
| `--stream-io` | direct | O_DIRECT reads for prompt chunks, so they do not push the hot experts out of the cache |

`STRATA_POOL_SPIN_US` (how long idle workers spin before sleeping, 20 ms by default) was tried at 100 and 300 us: no
gain from a cool start.

## Heat and turbo

The laptop heats up quickly under the model, and a hot CPU slows itself down in uneven steps. Two settings keep that under control.

**The temperature limit.** The server reads the CPU package temperature (`coretemp` or `k10temp` in sysfs) and the GPU's, once
a second. At the limit (80 C by default) it stops the engine process with SIGSTOP for a part of every half second and
resumes it with SIGCONT: a stopped process draws no CPU and starts no GPU work, so both cool. The step looks ahead: a rising
sensor counts as the temperature it will have in 3 seconds, judged by its own slope over the last 4 seconds (every sensor keeps
its own history, and a sensor that drops out starts a fresh one), so the pause starts before the
limit is crossed; the package keeps heating for a moment after the pause begins. When that projected temperature reaches the
limit the share grows by 10% plus 4 points per degree over it, each second, up to 85%; below the limit minus 3 degrees it
shrinks by 5 points a second, and inside that band it holds. Each change of 20 points and each start and end of the pausing is
one line in the log
(`[strata] thermal: cpu 83 C ...`), and the share and the temperature are in the `hardware` series of `GET /metrics`
(`thermal_duty`, `thermal_temp`). The engine is resumed when the server stops, and the kernel kills it if the server is
killed (`PR_SET_PDEATHSIG`), stopped or not. Where a process cannot be paused (Windows) the limit is not enforced and the
server says so. Checked with unit tests, with a busy process (it got less than half the CPU time and ran freely again after the
stop) and with the server's config wiring on the mock engine. With the real engine under a full load the look-ahead held the CPU
package at 78 C for the 80 C limit (three runs of eight 900-token requests; without it the package peaked at 82 to 87 C). The
per-sensor histories came after that run: one run of ten requests peaked at 79 C. Two variants that looked safer on paper
overshot instead and were dropped: forgetting the history after a slow sample (peaks of 82 and 83 C in two runs) and giving
the time back slowly inside the last degrees under the limit.

**The turbo boost.** `tools/turbo status|on|off` reads and writes `intel_pstate/no_turbo` (or `cpufreq/boost` on other drivers).
On this machine the CPU runs at up to 4800 MHz with the boost and 3300 MHz without. The switch belongs to root; a tmpfiles rule
(kept in the machine's restore repository) turns the boost off at boot and makes the file group-writable for `wheel`, so
neither `tools/turbo` nor `strata-serve --turbo` needs sudo. Without that rule `tools/turbo` asks for sudo, and the launcher only
warns. The boost is a machine-wide setting, not the model's: every program is slower without it. Speeds in this document were
measured with the boost on.

**The frequency cap.** Reading memory barely heats the CPU (four threads read 55.6 GB/s for two minutes with the package at 56 to
61 C); the heat comes from the arithmetic of the expert kernels, and it falls faster than the speed when the highest CPU
frequency, and the voltage with it, is lowered. Sustained speed of the IQ4_XS file with MTP under the 80 C limit, turbo off,
`STRATA_POOL_SPIN_US=0`, eight requests of 900 tokens after two warm-up requests (one or two runs each, on one machine):

| setting | tokens/s (median) | peak temperature |
|---|---|---|
| no cap, 2 workers | 11.0 to 14.8 | 77 to 79 C |
| 2.2 GHz, 4 workers | 28.6 and 27.2 (two runs) | 73 and 76 C |
| 2.4 GHz, 4 workers | 28.9 | 78 C |
| 2.6 GHz, 4 workers | 25.3 (the pauses of the governor return) | 78 C |
| 2.0 GHz, 6 workers | 28.8 | 74 C |
| 2.2 GHz, 6 workers | 31.1 and 29.8 (two runs) | 77 and 78 C |
| 2.4 GHz, 6 workers | 24.9 (the pauses return) | 80 C |

The default profile `auto` holds the cap by itself: `strata-serve` runs with `--cpu-max-mhz 2200`, 6 workers and no pool spin (the
setting of the table, `STRATA_POOL_SPIN_US=0`) and puts the old limits back when it stops (Ctrl+C and SIGTERM included; after a
`kill -9` run `tools/cpucap reset`). `--cpu-max-mhz 0` turns the cap off, the profiles with a memory budget (`lean` to `max`) run
without it, and 2200 MHz is two thirds of this laptop's 3.3 GHz, so another CPU needs its own value. The Q4_K_XL file at the same setting gave 29.4 tokens per second with a peak of 74 C, within 5 percent of IQ4_XS, so the file is to be chosen by memory (Q4_K_XL wants about 23 GB of free RAM, IQ4_XS about 18 GB) and quality, not by speed. `tools/cpucap status|set MHZ|reset` does the same by
hand. The limits belong to root. A tmpfiles rule makes them group-writable for `wheel`, the way the turbo switch is:
`z- /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq 0664 root wheel -` (kept in the machine's restore repository);
without it the server says which limits it could not write and runs without the cap. The cap is machine-wide like the turbo
switch: every program is slower while the server runs. The speed of one setting differed by up to a quarter between the night
and the morning; the repeats of the two best settings came within 5 percent of the first runs.
A package power limit did not help: with RAPL PL1 at 22 and 18 W, 6 workers and no frequency cap, the speed was 10.2 and 10.4 tokens per
second, as without any limit, because the hardware still runs the CPU at full frequency for short bursts and the governor has to pause
the engine. The whole model on the integrated GPU (llama.cpp with Vulkan) gave 10.8 to 11.3 tokens per second with the package at
59 to 65 C, and 8.8 with the discrete card added by a layer split.

## Measuring the cooling

Cooling is physical: software can only measure it and switch the machine's own cooling profile. `tools/cooling_kit.py` is the
protocol for that. For each CPU frequency cap (default 2.2, 2.4, 2.6 and 2.8 GHz) and each platform profile it runs the standard
benchmark through `tools/quant_bench.py` (6 workers, `STRATA_POOL_SPIN_US=0`, the server's own 80 C limit, requests back to back
without a cool-down between them), cools the package to 55 C before every step, and samples every 2 seconds the package
temperature, the hottest core, every CPU's frequency, the package power, the three fan speeds and the platform profile. A step
stops when the package reaches the limit, when its duration is over, or when the benchmark ends; the kit then puts back what it
changed (the cap files, and the platform profile when it switched it) and says in the result what it restored.

```sh
# check the sensors for 20 seconds, no benchmark
python3 tools/cooling_kit.py --sample-only 20
# the plan and the exact commands, nothing runs
python3 tools/cooling_kit.py --dry-run --setup "laptop on stand, fans cleaned"
# the measurement (four steps of 5 to 9 minutes each, plus the cool-downs)
python3 tools/cooling_kit.py --setup "laptop on stand, fans cleaned" --duration 300
# also compare the platform's cooling profiles (asks for the password in a dialog for every switch)
python3 tools/cooling_kit.py --setup "..." --profiles balanced,performance --apply-profile
```

`--setup` is a free-text label of the physical conditions (stand, cleaned fans, room, a cooling pad) kept in every result, so two
runs can be compared with the changed condition in view. The results go to `run/experiments/cooling/<time>/`: `samples.csv`
(one row per sample and step), `result.json`, `table.txt` and `chart.svg` (`--png` also renders a PNG). The one-screen table has,
per step, the sustained speed (the median of the second half of the requests, the heat-soaked state) next to the first request's,
the seconds until the package first reached 75 C and 80 C, the mean and the peak package temperature and the CPU fan's mean and
maximum RPM; `result.json` also has the times counted from the start of the first request (the model loading is not heat) and the
mean and peak power.

What the kit does not pretend:

- **The fan sensors.** The fan driver on this laptop (`bitland_mifs`, hwmon `fan1` CPU, `fan2` GPU, `fan3` SYS) read 0 RPM while the
  package was at 70 C. A fan that reads 0 for the whole run is reported as an unreliable sensor (idle fan or dead sensor, and "most
  likely not wired" when the package got hot), never as a stopped fan, and its columns show `n/a`. `--sample-only` shows this in
  seconds.
- **Power.** The RAPL counter (`/sys/class/powercap/intel-rapl:0/energy_uj`) is readable by root only here. The kit says so and
  prints the command that opens it until the next reboot (`sudo -A chmod a+r ...`); it never runs sudo on its own.
- **The platform profile** (`/sys/firmware/acpi/platform_profile`, here `low-power balanced balanced-performance performance`)
  belongs to root. Asking for a profile other than the current one without `--apply-profile` stops the kit and prints the command
  to paste. With `--apply-profile` the kit runs `sudo -A tee` for each switch (the password dialog names the reason) and puts the
  first profile back at the end, also after Ctrl+C or a failed step.
- **The 80 C limit** stays enforced by the server's own governor (it pauses the engine near the limit), so a step that holds 79 C
  shows lower speed rather than a higher temperature; the time to 80 C is then empty. The kit's own stop at the limit is the second
  guard. The duration counts from the start of the step, with 240 seconds for the model to load added.

## Staying in the background

The model keeps three or four cores busy for as long as it generates. At the default priority the browser, the editor and the
compiler compete with it on equal terms, and the desktop slows down. `strata-serve` therefore starts the server in the background
class (`--priority normal` turns that off):

- **A control group of its own in the idle mode.** The launcher puts the server in a transient systemd scope with
  `CPUWeight=idle` (the idle mode of the CPU controller: weight 1 and `cpu.idle=1`, so the group runs only on CPU that no other
  group wants) and `IOWeight=1`, next to the memory limit of the profile. A group of its own is needed: the scheduling class
  below orders tasks only inside one group, and a program started from a terminal otherwise shares the weight of that
  terminal's group with the browser's. The scope needs the `cpu` and `io` controllers delegated to the user manager (Fedora does
  it). The launcher starts a throwaway scope first and reads the weight back, and says so when the weight does not apply.
- **The idle scheduling class and the idle I/O class** (`SCHED_IDLE`, `ionice -c3`), set on the launcher itself, so the server and
  the engine inherit them.

On the laptop that was measured on, with a generation running, moving a running server into such a scope took the share of
time that other programs waited for the CPU (`/proc/pressure/cpu`, `some`) from 68% to 7% within seconds, while the engine kept
using the cores that nothing else wanted. The speed of the model under real competition was not measured: it goes down by
what the other programs take, and while they keep every logical CPU busy the server may stall altogether: a request waits, and
the temperature limit's resume signal is delayed until the server gets the CPU again. It carries on when they ease. During a
test session the model's own log showed one answer at 5.6 tokens a second instead of 36 while a test suite ran beside it.

What it does not do: the two threads of a core share their execution units, so a busy engine thread still slows the program on
the sibling thread a little (`taskset -c 0-3` keeps the engine to four logical CPUs and leaves four for the rest, at about
half the speed); it does not shrink the memory the model keeps in the page cache, so with little free memory the kernel swaps
other programs out, which is what `--profile` and its memory budget are for; and a server started by hand with
`serve/server.py`, not through `strata-serve`, keeps the normal priority. A server that is already running keeps its priority until it is restarted
with the launcher.

## What fits in 4 GB of VRAM

| context | chunk | `--mmq-group` | `--att-scores` | result |
| --- | --- | --- | --- | --- |
| 32768 | 512 | 16 | 8 | fits (0.49 GiB left) |
| 32768 | 768 | 16 | 8 | fits (0.21 GiB left before the scratch buffers were shared, so more now); a 3,099-token prompt in 15.5 s against 18.3 s at chunk 512 in the same session |
| 32768 | 1024 | 16 | 8 | did not start before the scratch buffers were shared (0.09 GiB left, then the engine exits); not tried since |
| 57344 | 512 | 16 | 8 | fits (0.21 GiB left); aborted after init with 0.08 GiB left before the scratch buffers were shared |
| 65536 | 512 | 16 | 8 | fits (0.14 GiB left with the tiled attention, 0.12 before it); out of memory at start before the changes below |
| 65536 | 512 | 16 | 4 | the default |
| 69391 | 512 | 16 | 4 | fits (0.11 GiB left), the largest window; 0.10 GiB left at `--att-scores 8` aborts after init |
| 73246 | 512 | 16 | 4 | not tried; needs about 44 MiB more |
| 24576 | 768 | 16 | 8 | fits, 123 tok/s prompt |
| 16384 | 768 | 32 | 16 | fits |
| 16384 | 1024 | 16 | 8 | aborts: no memory left to load a kernel module |

## Where the time goes (one decode token, whole model cached)

Per layer, about 1.0 ms before the fixes and 0.7 ms after (the engine's own counters, `CPU path ms`):

| Stage | before | after |
| --- | --- | --- |
| GPU: attention or recurrent layer up to the router | 0.27 ms | 0.27 ms |
| CPU: gate/up of 8 experts | 0.41 ms | 0.26 ms |
| CPU: down of 8 experts | 0.28 ms | 0.16 ms |
| job building and glue | 0.03 ms | 0.02 ms |

The GPU part is the dense weights (2.1 GiB read per token); judging by the kernel durations in an `nsys` trace they run at roughly two thirds of the card's memory bandwidth. The CPU part
reads 15.7 MB per layer; four threads read 54 GB/s at best and ggml's Q4_K dot gives 40 GB/s from DRAM on its own, 47 with
the row prefetch that is now in the loops (`kPrefetchRows` in `src/kernels/cpu/native_expert.cpp`).

## Waiting for the first token in pi

A question to pi sends about 1.5K tokens (system prompt 2.6K characters, four tools 3K characters, the question). Measured
through the real server, each question in a new pi session, same question:

| | read by the engine | pi wall time |
| --- | --- | --- |
| before: no checkpoints, O_DIRECT streaming | 1546 tokens in 18.1 s (85.6 tok/s) | 21.4 s every time |
| page-cache streaming, first session in a directory | 1546 tokens in 11.5-12.0 s (129-137 tok/s) | 15.5 s |
| + the CPU computes 96 of 256 experts per layer, first session after a reboot (model not in the page cache) | 1599 tokens in 10.8 s (148 tok/s) | 15.3 s (4.1 s of it the answer) |
| + the CPU share, model in the page cache | 1610 tokens in 8.8 s (183 tok/s) | the read, then the answer |
| first session in another directory (second checkpoint, 64 tokens earlier), with the CPU share | 87 tokens in 1.3 s | 4.4 s |
| any later session in a directory seen before (checkpoint in memory), with the CPU share | 19-26 tokens in 0.28-0.42 s | 0.8-0.9 s for answers of 2-10 tokens |
| first session after a server restart (checkpoint from disk) | 22 tokens in 0.58 s | about 4 s |

Two causes. (1) The recurrent layers cannot step back, so two requests that share only a prefix were read in full: the
engine now keeps a checkpoint of the state (recurrent layers, K/V cells, last hidden vector; 84 MB for 1.5K tokens) at
the start of the last turn, found as the second to last chat-template turn token (`--ckpt 6 --ckpt-token ID`), in pinned
host memory and in `run/ckpt` (`--ckpt-dir`). Restoring takes milliseconds. A restored run and a fresh run produce the
same bytes (a 1073-byte greedy answer, same checksum, fresh, from memory and from disk after a restart). (2) A prompt
chunk streams every expert (18.7 GB) through the card once; O_DIRECT reads ignore the page cache and run at the SSD's
3.4 GB/s even when the model is cached, so `auto` and `max` stream through the page cache instead (88 -> 140 tok/s on
3099 tokens). That is the PCIe link: the card gets 6.1 GB/s (PCIe 4.0 x4), 3.07 s per pass of 512 tokens, 167 tok/s at
best when the card computes every expert. `ExpertReader::begin` also waits for the card's last copy out of each slot before
reading into it again.

Two checkpoints are taken per new system prompt, one at the turn boundary and one 64 tokens earlier: pi writes the working
directory at the end of its system prompt, and the earlier checkpoint is still shared by a session in another directory.
The text of an answer can differ slightly from a run with other chunk boundaries (checkpoint positions change the chunking,
as `--chunk` does); a restored run equals the fresh run that made the checkpoint, byte for byte.

The CPU share lifts that ceiling (`--cpu-experts N`, the launcher sets 96 for `auto` and `max`). Of the 256 experts of a
layer the card gets the first 160 over PCIe and the CPU pool computes the last 96 from the mapped file with the AVX-512
kernels, at the same time; the CPU rows are copied to the card and merged with the card's rows. The share is a multiple of
`--mmq-group`. Engine alone, a 1546-token prompt, model in the page cache:

| `--cpu-experts` | 0 | 64 | 96 | 128 | 160 |
| --- | --- | --- | --- | --- | --- |
| prompt read | 10.85 s | 8.66 s | 7.65 s | 7.80 s | 8.88 s |
| tokens/s | 142 | 178 | 202 | 198 | 174 |

More CPU share stops paying when the CPU takes longer than the transfer. A 3099-token prompt: 22.1 s before, 15.2 s with 96
(204 tok/s), and all 16 greedy tokens equal the llama.cpp reference. The default path (`--cpu-experts 0`, used by the
capped profiles with O_DIRECT) is unchanged: 6 of 6 reference prompts match llama.cpp. A fresh, a memory-restored and a
disk-restored pi answer have the same checksum, and none of the 8 pi requests run with the share raised an engine error.
The CPU runs hotter during the read; its energy per token was not measured.

## AVX-512 kernels and one barrier per layer

`src/kernels/cpu/kq_avx512.cpp` computes Q4_K gate/up and Q5_K down rows for 1-8 tokens at once. ggml's AVX2 integer sums
are exact, so the kernel only keeps the 8-lane structure of the final sum and repeats ggml's floating-point steps in
ggml's order: every result is bit-identical (`ninja kq512_parity && ./kq512_parity --bench`: 200 cases, 0 mismatches).
The pool (`mode 7` in `src/kernels/cpu/pool.cpp`) cuts each expert into 2 gate/up and 4 down chunks; the thread that
finishes the last gate/up chunk quantizes the hidden values and releases that expert's down chunks, so a layer has one
barrier instead of two. `STRATA_NO_KQ512=1` and `STRATA_NO_FUSED=1` switch the two changes off.

Kernel, GB/s of weights, 4 threads, data outside the CPU cache (1 / 2 / 3 tokens): Q4_K ggml 44.6 / 26.7 / 19.7, AVX-512
58.2 / 42.0 / 31.9; Q5_K ggml 38.5 / 26.5 / 19.5, AVX-512 50.0 / 38.3 / 32.4.

Engine, 900 tokens of real code, warm cache, start at 55 C or less, three runs each in rotated order: before 33.3 / 33.9 /
36.3 (mean 34.5), AVX-512 only 41.2 / 36.1 / 35.3 (37.5), AVX-512 and one barrier 42.7 / 42.6 / 42.4 (42.6, +23%). The
generated tokens are identical in the three modes (same checksum over 600 tokens) and match llama.cpp on the parity prompts.
Through the real server without a memory limit: 32.1 / 32.0 before, 36.2 / 36.9 / 39.1 after. The capped profiles were
not measured again.

## IQ3_S and IQ4_XS kernels (opt-in)

The experts of an Unsloth UD-IQ4_XS file are mostly IQ3_S (gate and up) with IQ4_XS down. ggml's IQ3_S dot and the older AVX-512
IQ3_S kernel look the 512-entry grid up with scalar loads (instruction-bound, 3.7 instructions per cycle); gathers do not help
on a CPU with the Downfall microcode (`gather_data_sampling: Mitigation: Microcode`). `src/kernels/cpu/iq3s_vbmi.cpp` replaces
the lookup by byte permutes (AVX-512 VBMI, two grid weights packed per byte) and takes the sign bits with a bit shuffle
(BITALG); `src/kernels/cpu/iq4xs_vnni.cpp` looks the IQ4_XS nibbles up with `vpshufb` and sums with VNNI. Both work for any
number of tokens, so a token's rows do not depend on the drafting. They are off by default: `STRATA_IQ_VBMI=1` turns them on,
`STRATA_NO_IQ_VBMI=1` always turns them off, and a CPU without AVX-512 VBMI, BITALG, VNNI, BMI2, FMA and F16C ignores the flag.

Synthetic experts (2048 by 512), four threads on cores 0 to 3, one token per expert, weights streamed from memory, the laptop
not idle (`ninja iq_vbmi_bench && ./iq_vbmi_bench --bench --threads 4`, and the engine's own dispatch for the last three rows):

| Experts | per second | read, GB/s |
| --- | --- | --- |
| Q4_K gate/up and down (Q4_K_XL) | 22.0 thousand | 38.9 |
| IQ3_S gate/up and IQ4_XS down, ggml and the older kernels | 9.6 thousand | 14.0 |
| the same with `STRATA_IQ_VBMI=1` | 25.1 thousand | 36.6 |

On one thread the IQ3_S gate/up of an expert takes 340 us with ggml and 115 us with the new kernel, the IQ4_XS down 72 against 34 us. At 4 and 8 tokens per expert (prompt chunks) the gain stays: 1.9 and 1.6 times for gate/up, 3.0 and 2.6 times for down.
The new kernels match ggml's dot to a relative 1e-7 for 1 to 12 tokens (`./iq_vbmi_bench` fails above 1e-6); they are not bit-identical
because the order of the float additions differs. The Q4_K_XL path does not use them. A grafted UD-IQ4_XS file (MTP block from
the Q4_K_XL, `tools/gguf_graft.py`) loads and answers with and without the flag, with the same 378 characters of greedy text; its
speed through the server is not yet measured on a free machine.

## Dense part in Q6_K (a file variant, opt-in)

The dense matrices (attention, GDN, shared expert, MTP projection, embeddings) are Q8_0 in every file and sit in video memory
(about 1.6 GB), where every decode step reads them again. Q6_K stores the same weights in 0.82 bytes instead of 1.06, 23 percent
fewer bytes, for a small speed gain that is expected to be 3 to 4 percent (not measured yet). The router, the norms, the GDN
`ssm_a`/`alpha`/`beta`/`dt`/`conv` tensors and all expert tensors are never touched. The switch is a different model file, so there
is nothing to turn on in the engine and the Q8_0 file keeps working unchanged.

```sh
python3 tools/requant_dense.py SRC.gguf DST.gguf --dry-run                    # which tensors change, the new size, the disk space
python3 tools/requant_dense.py SRC.gguf DST.gguf --groups attn,ssm,shexp      # the default groups
python3 tools/requant_dense.py SRC.gguf DST.gguf --groups all                 # also output.weight, token_embd, the MTP projection
python3 tools/requant_dense.py --selftest                                     # the quantiser against ggml's reference, seconds
```

The groups are sets of tensor-name patterns: `attn` is `attn_q/k/v/output`, `ssm` is `attn_qkv`, `attn_gate`, `ssm_out` (the GDN
layers), `shexp` is `ffn_gate/up/down_shexp`, `head` is `output.weight`, `embd` is `token_embd.weight` and `mtp` is
`nextn.eh_proj`. Only Q8_0 tensors whose row length is a multiple of 256 are converted; a selected tensor of another type (the head
of the IQ4_XS file is Q6_K already) is listed and kept. The tool reads and writes one tensor chunk at a time, so memory stays small,
keeps every metadata entry and every other tensor byte for byte, and writes `DST.part` first. Run it with the system `python3`
(the project venv has no numpy), and keep the output inside `models/qwen3.6-35b-a3b/`: the launcher takes the tokenizer from there.

**Engine support (read from the code, not run).** Q6_K dense matrices already work on every GPU path, so no engine change was
needed: the loader accepts ggml type 14 (`native_mmvq_supported`); a decode step of 1 to 8 rows runs `native_mmvq` case 14 (the
8-column launches); a prompt chunk of more than 8 rows runs `Gemm::native`, whose dequantisation to FP16 has case 14; the embedding
gather accepts type 14; the head takes the whole-vocabulary path (`head_rows`). What a Q6_K matrix does not get are the Q8_0-only
fast paths: the lossless packed plane copy (`STRATA_Q8_PACKED`), the one-warp-per-row kernel for the small matrices (attention k and
v, the shared expert), the fused shared-expert gate and up kernel, and the draft head's subset kernel. Q6_K has more arithmetic per
byte, so the gain can be smaller than the byte saving, or negative: compare the step times in the server log, not only the speed.

**Quality.** The weights are re-quantised, so the answers are not identical to the Q8_0 file's. The quantiser follows ggml's
`quantize_row_q6_K_ref`; `--selftest` prints its agreement with that reference and the relative RMS error of Q8_0 to Q6_K on random
data. Judge the file by the perplexity of the llama.cpp perplexity tool on the source file and on the variant over the same text.

## Against llama.cpp

llama.cpp b3fa6995 on the same machine (`-ngl 99 --n-cpu-moe 99 -t 4 -fa on`, experts on the CPU, no drafting), greedy
decoding of the same code prompt, measured on the third and fourth request of one server under the same cgroup budget:

| budget | llama.cpp | this build (greedy + MTP) |
| --- | --- | --- |
| 16G | 21.1 / 24.5 | 25.4 |
| 22G | 27.4 / 27.3 | 33.1 |

Prompt reading, 3099 tokens, 9G budget: llama.cpp 49 tok/s, this build 84-86 (chunk 512), 123 (chunk 768, 24K window).

## Where the ceiling is

One generation step with MTP checks 3 tokens and yields 2.7 on average (84% of the drafts accepted). Counters of a 400-token
run on a cool machine, 22G budget, whole model cached (75 ms per step, 35.6 tok/s):

| Part of a step | Measured | Memory ceiling | Note |
| --- | --- | --- | --- |
| GPU, dense weights up to the router | 14.9 ms | 13.2 ms | 2.27 GB per step at 172 GB/s, the card's measured read speed |
| CPU, routed experts | 52.0 ms | 26.1 ms | about 1.4 GB per step at 54 GB/s, the machine's measured read speed; achieved 29 GB/s |
| drafting, sampling, glue | 8.5 ms | - | |

60 tok/s would need 45 ms per step. The two ceilings added (the GPU and the CPU wait for each other layer by layer) give
39 ms, about 68 tok/s, with no other loss, so 60 is out of reach in practice. The same run repeated right after on a hot
CPU spent 11.7 s instead of 7.7 s in the expert part (+51%) and the same 2.2 s on the GPU part: 25.9 tok/s.

Bandwidth figures behind this: VRAM read 171.7 GB/s; RAM read 18.3 / 34.0 / 46.5 / 54.3 GB/s with 1 / 2 / 3 / 4 threads;
ggml's Q4_K dot product 18.6 GB/s per thread from cache, 40.7 GB/s with four threads from DRAM, 47.7 with the row
prefetch; an expert's weights are 1.9 MB, the pool reads 1.42 tokens' worth per weight set in an MTP step.

Untried, with the estimate in brackets (a calculation, not a measurement): one barrier per layer with whole experts per
thread (+5-10%); AVX-512 VNNI kernels that decode a weight once for several tokens (+10-20%); starting the next draft on
the GPU while the CPU verifies (up to +15%); 3-bit experts (+20-30%, changes the model's output); 6-bit dense GPU weights
(+5%, changes the output).

## Long context and video memory

A token of the window costs 11,968 bytes of q8 K/V cache (11 attention layers x 2 K/V heads x 256 x 2 for K and V) plus
about 1 KB of buffers that scale with the window (the FP16 copy of one K / V head for the prompt attention, the attention
partial sums), about 13 KB in all. A 128K window would need 1.46 GiB of K/V alone on a card whose dense weights take
2.03 GiB; 64K fits, as below.

**The decode attention kernel.** The old kernel ran one block per query row and head, so the 8 query heads that share a
K/V head each read the same cache chunk, and an MTP verify step (3 rows) read it three times more. `k_attn_part_gqa`
runs one block per (chunk, row, K/V head) and serves all 8 heads from one read of the chunk. Its arithmetic and the
order of summation are the old kernel's (8 lanes per cell add the q8 block sums in block order; the softmax uses the
same reduction trees), so the output is bit for bit the same: a temporary comparison of both kernels on real prompts (1.5K
and 14K tokens, MTP 0 and 2) found 0 differing values in 27-275 million, and the token checksums of the final build equal
those of the compared build. `STRATA_Q36_ATTN_PER_HEAD=1` keeps the old kernel. Measured with a cooled CPU (start at or below 55 C), a
32K window, the cache already filled by a first request:

| | before | after |
| --- | --- | --- |
| kernel time per call, 1.5K fill, MTP 2 (nsys) | 246 us | 56 us |
| kernel time per call, 14K fill, one token (nsys) | 931 us | about 250 us or less |
| decode without MTP at 1.5K / 14K / 28K fill, ms per token | 35.1 / 47.6 / 55.0 | 32.9 / 36.5 / 38.8 |
| MTP 2 step at 28K fill, ms | 141.6 and 147.5 | 84.8 |
| MTP 2 step at 1.5K fill, medians of six requests per process, ABBAAB order, ms | 69.8, 75.6, 76.3 | 64.5, 65.8, 83.0 |

The cost of a full 32K window on an MTP step falls from about 70 ms (nearly double) to about 15-20 ms; at a short
context the speed is the same within the run-to-run noise (+-15%, the CPU's thermal state matters more than this change).
Caveats: the first runs of this series put the new kernel behind the old one at 1.5K; interleaved runs and per-kernel
profiles show that was interference. The 14K run without MTP stopped after 11 tokens; the 28K step is a single run. Nsight
Compute needs root here (ERR_NVGPUCTRPERM), so the remaining gap to the memory roof (about 60 GB/s of unique K/V at 14K
against 171 GB/s) is not broken down by stall reason.

**Where the video memory goes.** An `LD_PRELOAD` wrapper around `cudaMalloc` logged every allocation (985 live ones at a
32K window); the call sites were resolved with `nm`. In MiB:

| | before | after |
| --- | --- | --- |
| dense weights (Q8_0 and others) | 1978.4 | 1978.4 |
| F32 weights, including the router | 101.8 | 19.8 |
| router with the shared expert's gate row (`router_ext`) | 82.3 | 82.3 |
| chunk and attention buffers (including the attention partial sums) | 441.1 | 328.2 |
| q8 K/V cache (22 buffers of 17 MiB) | 374.0 | 374.0 |
| expert staging (2 x 35.25), recurrent state (56), CPU-path result buffer, the rest | 199.4 | 175.4 |
| **total** | **3177.1** | **2958.2** |

Three things were taken out, none of which touches the arithmetic: the attention partial sums were sized for 1,024 query
rows (16 heads x 64 tokens) while the split-K path only takes up to `kGemmT` = 8 tokens (128 rows; prompt chunks go through
GEMMs), 112.9 MiB; the F32 router copy duplicated the first 256 rows of `router_ext`, 82.0 MiB; the CPU-path result buffer
was sized for a whole chunk while a layer sends at most `hybrid_t` = 128 rows through it, 24.0 MiB. Free memory after
init at a 32K window went from 0.14 to 0.33 GiB, a 48K window now starts, prompt reading speed is unchanged (195-205 tok/s
at 1.5K). Outside `cudaMalloc` the CUDA context, cuBLAS and kernel code take about 0.6-0.8 GiB. One discrepancy is open: the
`nvidia-smi` peak fell only from 3743 to 3725 MiB although the engine's free-memory counter rose by 0.19 GiB; the driver
probably takes part of the freed memory after init (kernel code, graphs), not checked.

**Two more changes that made 64K fit**, again without touching the arithmetic. (1) The prompt attention ran its K / V
conversion, score buffers and softmax for both K / V heads at once; it now runs one K / V head at a time (the same GEMMs
with the same shapes, the same softmax kernel), so `kv16`, `att_s` and `att_p16` hold one head: 40.8 MiB less at 32K,
57.5 MiB at 64K. (2) The scratch buffers of a GDN layer's mixer (`qkv`, `hc`, `z`, `gdn_y`), of an attention layer's mixer
(`qg`, `kx`, `vx`, `q`, `gate`, `att`, `q16`, `kv16`, `pacc`) and of the MoE block (`eparts`, `moe_out`, the shared
expert's buffers, the MMQ activations and intermediates) are never live together on the stream, so they now share one
region as large as the largest of the three: 78 MiB instead of 204 MiB at 32K, 110 MiB instead of 236 MiB at 64K.
`commit()` uses `hc` and `z` between decode steps, when no layer runs. Live `cudaMalloc` total: 3177.1 MiB at the start,
2958.2 after the first three changes, 2792.2 MiB at a 32K window now (3198.1 MiB at 64K).

| window | free after init, at the start | after the first three changes | now |
| --- | --- | --- | --- |
| 32768 | 0.14 GiB | 0.33 GiB | 0.49 GiB |
| 49152 | not measured | 0.13 GiB | not measured |
| 57344 | not measured | 0.08 GiB, aborts after init loading a kernel module | 0.21 GiB |
| 65536 | out of memory at start | out of memory at start | 0.12 GiB, runs |

0.08 GiB is not enough (the engine aborts when it loads the first kernel modules), 0.12 GiB is, so about 0.1 GiB is the
floor, and with the tiled attention below the ceiling of this card in q8 is 69,391 tokens. Checks of the build with all five changes, all at the same output: the
96-token checksums at 1.5K and 14K fill with MTP 0 and 2 equal the bit-checked ones (`c33f0af2b4c9`, `c2b9567fba35`), llama.cpp
parity 6 of 6, 12 prompt shapes (5 to 1,100 tokens, checkpoints on) complete at a 65,536 window, and a 40,891-token prompt
at a 65,536 window gives the same 64 tokens with MTP 0 and MTP 2 (`5c9423fabb62`). Reading that prompt takes 362 s (113
tok/s on average: the attention over a long context costs more per token), decoding after it takes 37.7 ms per token
without MTP and 102.5 ms per MTP step.

**Tiled prompt attention** (the third change that moved the window). Up to 4,096 keys (`STRATA_Q36_FLASH_MIN`) a prompt chunk
keeps the row-wide softmax, bit for bit what the engine always computed. Beyond that its attention runs in tiles of keys
(1,024 at `--att-scores 8`, 512 at 4), one K / V head at a time: the tile of K is converted to FP16 once, one GEMM scores it
against the whole query tile, `k_attn_online` takes the tile's maximum, rescales the output rows accumulated so far, adds the
tile's sum to the running one and writes the FP16 weights exp(s - running maximum), the tile of V is converted, a second
GEMM adds weights x V into the output (beta = 1), and `k_attn_norm` divides the rows by the running sum at the end. The old
code converted the whole K and V again for every slice of 8 to 128 queries, which is what made long prompts slow, and held a
FP16 copy of the whole context (33.5 MiB at 64K): `kv16` now holds one tile (4,096 rows, 2 MiB) and the scratch arena
shrinks from 110 to 80 MiB at 64K.

The rounding differs from the row-wide softmax (the weights are rounded before they are divided by the sum), so above 4,096
keys the output is not bit-identical. Both paths on the same inputs, every attention layer and chunk: 47.0 million values of
a 1.5K prompt (tiled path forced from 512 keys) differ by at most 4.3e-03 where the values reach 15.6, 1.4e-04 of the rms of
the values; at a 14K fill (404 million values) at most 6.2e-03, 1.0e-04. llama.cpp parity is 6 of 6 with the tiled path forced
on for every chunk with more than one key. On the synthetic 1.5K prompt the 96 generated tokens then diverge from the third
token on (near-tied logits; checksum 090e5f7a2590 against c33f0af2b4c9), after the 41K prompt the 64 tokens differ too
(8d3eb7e99100 against 5c9423fabb62). Reading speed, prompt only: 14K tokens in 71.4 and 70.5 s against 81.3 and 73.2 s
(ABBA order, about 8% less); 40,891 tokens at a 65,536 window in 278 s (147 tok/s) against 362 and 348 s (113 tok/s), 20%
less; decode after it 86.2 ms per MTP step. `--att-scores 4` (now the default) keeps the 96-token checksum of the 1.5K prompt
(`c33f0af2b4c9`) and costs no speed that the run-to-run noise (up to 10%) shows.

Free after init: 0.14 GiB at 64K with `--att-scores 8` (0.12 before), about 0.15 GiB with 4. The window can go to **69,391**
tokens with `--att-scores 4` (0.11 GiB left, runs; 0.10 GiB at 8 aborts after init): the K / V buffers are allocated in 2 MiB
steps, 65,536 is a step boundary, and every size above it up to 69,391 costs 44 MiB at once (22 buffers x 2 MiB), so
66,560 or 67,584 only waste memory; the next step, 73,246, needs about 44 MiB more.

**What else could be squeezed** (calculations, not measurements, unless said):

| idea | gain at 32K | cost |
| --- | --- | --- |
| dense weights Q8_0 to Q6_K / Q5_K | about 0.45 / 0.68 GiB | changes the output (requantisation), a little faster |
| K/V cache q8 to q4 | about 180 MiB (0.73 GiB at 128K) | changes the output |
| a K/V window only for the MTP block | about 34 MiB at 32K, 68 MiB at 64K | output unchanged (the main model verifies); fewer accepted drafts at a long context |
| prompt MMQ group 16 to 8 (staging 70.5 to about 35 MiB) | about 35 MiB | not measured: more, smaller copies and MMQ launches |
| smaller cuBLAS / cuBLASLt workspaces, stack and heap limits | unknown (the context, cuBLAS and kernel code take 0.6-0.8 GiB) | may change cuBLAS's algorithm choice and with it the rounding |
| prompt chunk 512 to 256 | about 100 MiB (estimate) | twice the expert passes over the bus; prefill much slower |
| output matrix (515 MiB, Q8_0) on the CPU | 515 MiB | read on every token, about +8-10 ms per step |
| old K/V in RAM, attention over it on the CPU | a 128K window | a large job; about +0.22 ms per step for every 1K tokens outside VRAM (11,968 B at 54 GB/s) |

The limit is 69,391 tokens now (65,536 is the default, with about 0.15 GiB left); the MTP window and the staging group
would add roughly 8K more tokens together (68 + 35 MiB at 13 KB per token), but only in steps of 44 MiB. 128K on this card
needs either lower precision (q4 K/V, lighter weights) or the old part of the cache in RAM.

## Ideas that were checked

Details, analogies and charts are in `docs/index.html`, section 6.

| Idea | Result |
| --- | --- |
| Another cache policy (LFU, pinning, oracle replayed on routing traces recorded with `STRATA_ROUTE_TRACE`) | LRU is within 2 points of the oracle at 12G and above; the oracle adds at most 9 points at 8G; pinning is worse |
| Compute cached experts first while the disk loads the rest | 14.05 against 14.65 tok/s at 12G: no gain, reverted |
| Predict the next layer's experts (its router on this layer's input) and prefetch them | prediction is good (77.5 / 88.3 / 94.5% of the real picks among 8 / 12 / 20 predicted) but 12G speed did not improve: 14.1 at 2-3 predicted, 13.3 at 5, 10.7 at 12, 7.9 at 20 (against 14.4); removed |
| n-gram lookup drafting | 1.69 tokens per step against 2.68 for MTP on generated code |
| Experts in the 143 MB of free VRAM | 75 experts cover 0.8% of the accesses |
| Whole working set in VRAM | 8,458 of 10,240 experts used within 600 tokens of one text |
| GPU reads the SSD directly | GeForce only gets cuFile compatibility mode (through host memory); the SSD link trained at PCIe Gen3 x4, which is the measured 3.4 GB/s |
| Spin threshold of idle workers under sustained heat (2000 tokens) | default 32.9 / 32.9, 100 us 28.0 / 30.4 tok/s |
| CPU computes a share of the experts of a prompt chunk while the card streams the rest | 142 -> 202 tok/s on a 1546-token prompt (96 of 256 experts on the CPU), above the 167 tok/s PCIe ceiling |
| Integrated GPU | shares the same memory and thermal budget; not benchmarked |
| Assembly | kernels sit near the memory roofs; the gap is in barriers and idle gaps |

Not measured (needs root, the password prompt was not answered): raw NVMe against the LUKS volume, CPU energy per token.

## Testing the levers

Five changes are meant to make the IQ4_XS file faster on this laptop without turbo, overclocking or going above 80 C. Each is
off by default, and with its switch off the engine behaves as before. `tools/lever_matrix.py` is the runner for testing them:
it reads one definition file per lever from `tools/levers.d/*.json`, plans runs and executes them one at a time through
`tools/quant_bench.py` with the thermal rules of this document (the 2200 MHz cap, 6 workers, `STRATA_POOL_SPIN_US=0`, the 80 C
limit, a wait for the package to cool to 55 C before every run). Nothing in this section has been measured yet.

| # | Lever | Switch | Needs | What proves it |
| --- | --- | --- | --- | --- |
| 1 | Better cooling | `tools/cooling_kit.py` (no engine switch; the platform profile belongs to root) | nothing | the kit's table: sustained tokens per second and the time to 75 and 80 C per frequency cap |
| 2 | Overlap GPU and CPU work in a step | env `STRATA_Q36_OVERLAP` and flag `--overlap` | nothing | identical greedy text, then speed |
| 3 | Experts split between the CPU and the Iris Xe iGPU | env `STRATA_IGPU_EXPERTS` and flag `--igpu-experts` | build with `-DSTRATA_IGPU_VULKAN=ON` | identical greedy text (or a stated numeric tolerance), then speed |
| 4 | Fewer active experts per token | env `STRATA_Q36_EXPERTS_USED` and flag `--experts-used` | nothing | not identical by design: a quality check, then speed |
| 5 | Dense part Q8_0 to Q6_K | a model file made by `tools/requant_dense.py` | the file (about 16.7 GiB of disk per variant) | not identical by design: perplexity against the source file, then speed |

The names of the switches of levers 2 to 4 are the ones in their own sections of this document and in `tools/levers.d/`; where
they differ, the definition file is the truth, and `python3 tools/lever_matrix.py list` shows what is on this machine.

**The order to follow.**

1. *Free the machine.* Close everything heavy, then `python3 tools/machine_state.py` must say it is quiet (turbo off, no engine or
   server running, low load, memory free). `tools/cpucap status` must show the limit files as writable.
2. *Cooling first.* `python3 tools/cooling_kit.py --sample-only 20` to see which sensors work, then the kit as described under
   "Measuring the cooling". Take the best cap from its table as `--cpu-max-mhz` for everything after, and keep the `--setup` text
   of the run. The cap is machine-wide and every later number depends on it.
3. *Build and make what the levers need.* The usual build for levers 2 and 4; a build with `-DSTRATA_IGPU_VULKAN=ON` for lever 3
   (the runner reads `build/CMakeCache.txt` and calls the lever unavailable without it); `tools/requant_dense.py` for lever 5.
   `python3 tools/lever_matrix.py list` says which levers are available and why not.
4. *Baseline and each lever alone.* `python3 tools/lever_matrix.py plan --each` shows the runs and their exact command lines;
   `python3 tools/lever_matrix.py run --each` executes them. The session starts and ends with a baseline run: the difference
   between the two is the drift of the machine (heat, other programs, the time of day), and the summary marks it `UNSTABLE` above
   5 percent, in which case the deltas are not to be trusted and the session is repeated later. A lever's delta is measured
   against the mean of the baselines.
5. *Parity.* The summary compares the first measured answer of every run with the baseline's (greedy, temperature 0, the same
   prompt): `identical`, `DIFFERS` or `n/a`. For the levers that must not change the output (2, and 3 within its stated
   tolerance) a `DIFFERS` is a bug, not a trade-off: stop and report it before looking at the speed. Levers 4 and 5 differ by
   design, so their parity column only shows how far the text drifted.
6. *Quality.* For levers 4 and 5 run the perplexity tool of llama.cpp (`llama.cpp/build/bin/llama-perplexity`, the `quality_check` of
   the definition) on the source file and on the variant over the same text, and read the answers of the two runs side by side.
   A speed gain is worth nothing if the quality loss is larger than is acceptable. The runner does not run quality tools: it
   prints the command and writes `not run` in the column.
7. *Combinations.* Only after the single runs: `python3 tools/lever_matrix.py run --levers overlap,experts_used` for the pairs
   that passed steps 5 and 6, `run --pairs` for all of them, and `run --all` for every available lever together. A pair can
   fail where its members do not (two switches setting the same environment variable with different values are refused at
   plan time, with both sources named).
8. *Decide.* Keep a lever only if its gain is above the drift, its parity or quality check passed, and its peak package
   temperature stayed at or below the limit.

Every command takes `--dry-run`: it prints the plan and the command lines and runs nothing. Results go to
`run/experiments/levers/<time>/`: one JSON per run (with the plan, the exact command and quant_bench's result), `summary.json`
and `summary.txt` (rewritten after every run, so an interrupted session keeps what it measured), `chart.svg` and `raw/` with
the benchmark's own files. A stop signal ends the running benchmark gracefully (it stops its server and puts the cap back), the
runner checks that the frequency limit files read as they did before the session, and lists any engine process left behind.

**The definition files.** One JSON file per lever, loaded by name:
`{"name", "lever": 1..5, "title", "kind": "engine" | "system" | "model-file" (default engine), "variants": [{"name", "env",
"engine_args", "launcher_args", "model_file"}], "needs": {"build_option", "model_file", "root"}, "quality_check", "parity_check",
"notes"}`. The first variant is the lever "on" (`--levers overlap`; another one is `--levers overlap:other`). `engine` levers are
measured with the variant's `env` and `engine_args` (`launcher_args` are only printed: the benchmark harness cannot pass launcher
flags); `system` levers are not run by the runner (the cooling kit does them) and never take part in `--each`, `--all` or
`--pairs`; `model-file` levers replace the model with the variant's `model_file` and are available only when the file exists. The
environment of a run is the baseline's (`STRATA_POOL_SPIN_US=0`) plus each selected variant's plus `--env`; the same variable with
different values from two sources is an error that names both, the same value twice is fine, and identical engine arguments are
passed once. `python3 tools/lever_matrix.py --help` prints the same schema.

## Benchmarking pitfalls

- The page cache is global: a new cgroup inherits pages that an earlier run loaded, so a cap means nothing until the model
  file is evicted (`posix_fadvise(DONTNEED)` on it, no root needed).
- `perf record -e cpu-clock:u -p <pid of strata-q36>` works with the default `perf_event_paranoid`; `pgrep -x strata-q36`
  finds the engine, a plain `pgrep -f` also matches `timeout` and the shell that runs it.
- The engine prints `CPU path ms` at exit: router wait, job building, gate/up, quantize, down.
