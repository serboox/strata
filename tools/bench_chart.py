import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

BUDGET_ORDER = ["8G", "12G", "16G", "22G", "none"]
SERIES = [("q4kxl_control", "Q4_K_XL (control)", "#8a8f98"), ("iq4xs_kernels_off", "IQ4_XS, old kernels", "#e08a2b"),
          ("iq4xs_kernels_on", "IQ4_XS, new kernels", "#2b6fd6")]
INK, GRID, MUTED = "#1c2430", "#d9dee5", "#5b6573"


def load(paths: list) -> list:
    results = []
    for path in paths:
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"bench-chart: {path}: {error}", file=sys.stderr)
            continue
        results.append(result)
    return results


def speeds(result: dict) -> dict:
    """The measured speeds, the best warm-up speed (the machine still cool) and the median of the measured ones."""
    rows = result.get("rows", [])
    measured = [row["tok_s"] for row in rows if not row.get("warmup") and row.get("stat_lines") == 1 and "tok_s" in row]
    warm = [row["tok_s"] for row in rows if row.get("warmup") and "tok_s" in row]
    ordered = sorted(measured)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (sum(ordered[len(ordered) // 2 - 1:len(ordered) // 2 + 1]) / 2 if ordered else None)
    return {"measured": measured, "burst": max(warm) if warm else None, "median": median}


def text(x, y, content, size=12, anchor="start", fill=INK, weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" '
            f'font-family="DejaVu Sans, Arial, sans-serif"{transform}>{html.escape(str(content))}</text>')


NAMES = {"tune_default": "default (3 workers)", "tune_workers2": "2 workers", "tune_spin50": "spin 50 us", "tune_spin0": "no spin",
         "tune_mtp3": "3 drafts", "tune_control": "Q4_K_XL control", "tune_default_again": "default again", "tune_w2_spin0": "2 workers, no spin",
         "tune_w1_spin0": "1 worker, no spin", "tune_control_w2_spin0": "control, 2 workers, no spin",
         "tuneg_default": "default, looking-ahead governor", "tuneg_w2_spin0": "2 workers, no spin, looking-ahead governor"}
PALETTE = ["#8a8f98", "#e08a2b", "#2b6fd6", "#2f9e5a", "#9b59b6", "#d62b2b", "#17a2b8", "#7f8c2b"]


def series_for(labels: list) -> list:
    """Bars for these run labels in this order, coloured from the palette; the known labels get their readable names."""
    names = {**NAMES, **{label: name for label, name, _ in SERIES}}
    return [(label, names.get(label, label), PALETTE[index % len(PALETTE)]) for index, label in enumerate(labels)]


def speed_chart(results: list, title: str, note: str, series: list | None = None) -> str:
    series = series or SERIES
    by_key = {(result.get("budget"), result.get("label")): speeds(result) for result in results}
    budgets = [budget for budget in BUDGET_ORDER if any(key[0] == budget for key in by_key)]
    top = max([value for item in by_key.values() for value in item["measured"] + ([item["burst"]] if item["burst"] else [])] + [10.0])
    top = (int(top / 10) + 1) * 10
    left, right = 64, 24
    legend_labels = [label for _, label, _ in series] + ["measured request", "first request, cool"]
    advances = [34 + int(6.6 * len(label)) for label in legend_labels]
    legend_rows, cursor = 1, left
    for advance in advances:
        if cursor + advance > 960 - right and cursor > left:
            legend_rows, cursor = legend_rows + 1, left
        cursor += advance
    width, height, upper, lower = 960, 420 + 20 * legend_rows, 70, 52 + 20 * legend_rows
    plot_w, plot_h = width - left - right, height - upper - lower
    group_w = plot_w / max(1, len(budgets))
    bar_w = group_w * 0.8 / len(series)

    def y_of(value):
        return upper + plot_h * (1 - value / top)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img"><rect width="{width}" height="{height}" fill="#ffffff"/>',
             text(left, 26, title, 17, weight="bold"), text(left, 46, note, 12, fill=MUTED)]
    for tick in range(0, top + 1, 10):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y_of(tick):.1f}" y2="{y_of(tick):.1f}" stroke="{GRID}"/>')
        parts.append(text(left - 8, y_of(tick) + 4, tick, 11, "end", MUTED))
    parts.append(text(18, upper + plot_h / 2, "tokens per second", 12, "middle", MUTED, rotate=-90))
    for g, budget in enumerate(budgets):
        gx = left + g * group_w + group_w * 0.1
        parts.append(text(left + g * group_w + group_w / 2, height - lower + 22, f"budget {budget}", 13, "middle", INK, "bold"))
        for s, (label, _, color) in enumerate(series):
            item = by_key.get((budget, label))
            if item is None or item["median"] is None:
                continue
            x = gx + s * bar_w
            parts.append(f'<rect x="{x + 3:.1f}" y="{y_of(item["median"]):.1f}" width="{bar_w - 6:.1f}" height="{plot_h * item["median"] / top:.1f}" fill="{color}" opacity="0.88"/>')
            parts.append(text(x + bar_w / 2, y_of(item["median"]) - 6, f'{item["median"]:.1f}', 12, "middle", INK, "bold"))
            for value in item["measured"]:
                parts.append(f'<circle cx="{x + bar_w / 2:.1f}" cy="{y_of(value):.1f}" r="3" fill="#ffffff" stroke="{INK}" stroke-width="1"/>')
            if item["burst"] is not None:
                cx, cy = x + bar_w / 2 + bar_w * 0.28, y_of(item["burst"])
                parts.append(f'<path d="M{cx - 5:.1f} {cy + 5:.1f} L{cx + 5:.1f} {cy + 5:.1f} L{cx:.1f} {cy - 5:.1f} Z" fill="{INK}"/>')
    items = [(f'<rect x="{{x}}" y="{{y}}" width="14" height="14" fill="{color}"/>', label) for _, label, color in series]
    items.append(("<circle", "measured request"))
    items.append(("<path", "first request, cool"))
    x, row = left, 0
    for (shape, label), advance in zip(items, advances):
        if x + advance > width - right and x > left:
            x, row = left, row + 1
        y = height - 14 - 18 * legend_rows + row * 18
        if shape.startswith("<rect"):
            parts.append(shape.format(x=x, y=y))
        elif shape.startswith("<circle"):
            parts.append(f'<circle cx="{x + 6}" cy="{y + 7}" r="3" fill="#fff" stroke="{INK}"/>')
        else:
            parts.append(f'<path d="M{x} {y + 13} l10 0 l-5 -10 z" fill="{INK}"/>')
        parts.append(text(x + 20, y + 12, label, 12))
        x += advance
    parts.append("</svg>")
    return "".join(parts)


def read_thermal(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        try:
            rows.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
        except (IndexError, ValueError):
            continue
    return rows


def thermal_chart(rows: list, title: str, limit_c: float = 80.0) -> str:
    width, height, left, right, upper, lower = 960, 340, 64, 24, 54, 44
    plot_w, plot_h = width - left - right, height - upper - lower
    count = max(2, len(rows))
    cpu_top = max([row[1] for row in rows] + [limit_c + 5])
    cpu_top = (int(cpu_top / 10) + 1) * 10

    def x_of(i):
        return left + plot_w * i / (count - 1)

    def y_cpu(value):
        return upper + plot_h * (1 - (value - 40) / (cpu_top - 40))

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img"><rect width="{width}" height="{height}" fill="#ffffff"/>',
             text(left, 24, title, 17, weight="bold")]
    for tick in range(40, cpu_top + 1, 10):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y_cpu(tick):.1f}" y2="{y_cpu(tick):.1f}" stroke="{GRID}"/>')
        parts.append(text(left - 8, y_cpu(tick) + 4, f"{tick} C", 11, "end", MUTED))
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y_cpu(limit_c):.1f}" y2="{y_cpu(limit_c):.1f}" stroke="#d62b2b" stroke-dasharray="6 4"/>')
    parts.append(text(width - right, y_cpu(limit_c) - 5, f"limit {limit_c:.0f} C", 11, "end", "#d62b2b"))
    for series, color in ((1, "#d62b2b"), (3, "#2b6fd6")):
        points = " ".join(f"{x_of(i):.1f},{y_cpu(row[series]):.1f}" for i, row in enumerate(rows))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    if rows:
        parts.append(text(left, height - 14, f"{rows[0][0]}", 11, fill=MUTED))
        parts.append(text(width - right, height - 14, f"{rows[-1][0]}", 11, "end", MUTED))
    parts.append(text(left + 120, height - 14, "CPU package", 12, fill="#d62b2b"))
    parts.append(text(left + 230, height - 14, "GPU", 12, fill="#2b6fd6"))
    parts.append("</svg>")
    return "".join(parts)


def svg_size(svg: str, default: tuple = (960, 460)) -> tuple:
    """Width and height from the viewBox of the SVG, so a taller chart is not cut off when it is rendered."""
    match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    return (int(match.group(1)), int(match.group(2))) if match else default


def to_png(svg_path: Path, png_path: Path, size: tuple = (960, 460)) -> bool:
    """Render an SVG with headless Chrome; False when Chrome is missing or fails."""
    try:
        done = subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", f"--window-size={size[0]},{size[1]}",
                               f"--screenshot={png_path}", f"file://{svg_path}"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and png_path.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the benchmark results (speed by memory budget) or a temperature log as SVG, optionally PNG.")
    parser.add_argument("files", nargs="*", type=Path, help="quant_bench.py result files")
    parser.add_argument("--thermal", type=Path, help="a thermal sampler log to draw instead")
    parser.add_argument("--title", default="Decode speed with MTP, 4 GB laptop GPU")
    parser.add_argument("--note", default="Turbo off, temperature limit 80 C, 900 generated tokens per request")
    parser.add_argument("--series", help="comma-separated run labels to draw as bars, in this order (default: the three standard ones)")
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    if args.thermal is not None:
        svg, size = thermal_chart(read_thermal(args.thermal), args.title), (960, 340)
    else:
        results = load(args.files)
        if not results:
            return 1
        series = series_for(args.series.split(",")) if args.series else None
        svg, size = speed_chart(results, args.title, args.note, series), (960, 460)
    args.svg.write_text(svg, encoding="utf-8")
    if args.png is not None and not to_png(args.svg.resolve(), args.png.resolve(), svg_size(svg, size)):
        print("bench-chart: the PNG could not be made", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
