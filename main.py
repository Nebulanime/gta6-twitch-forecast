"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Model (fit fresh from each case's own comparables, at run time):

  Every past big launch in the case has an hourly viewer curve around its
  release day.  For hour h of the release day, look at the log-ratio

      L_h = ln(viewers_h / viewers_1)

  across the comparables.  Two facts the data shows (37-41 launches):

  1. The launch-day shape is a dip then a ramp: the median curve falls to
     ~0.75x hour 1 in the deep night (UTC hours 3-5) and climbs to ~2.4x by
     hour 19-20 as the Americas unlock and prime time arrives.
  2. The ramp depends on how big hour 1 already is: the log-ratio at late
     hours falls almost one-for-one with ln(hour-1 viewers) (slope about
     -0.9, R^2 about 0.88).  In absolute terms the day's peak is nearly
     scale-free across launches -- a launch already hot at 00:00 UTC has
     little headroom left, a quiet one ramps hard.  A second predictor, the
     hour-1 channel count, adds a little more (holding viewers fixed, a
     more fragmented audience ramps slightly less).

  So for each hour h we fit, over the comparables,

      L_h ~ a_h + b_h * ln(v1) + c_h * ln(ch1)

  by least squares, predict the case's own L_h from its hour-1 print and
  channel count, and set the median to p1 * exp(L_h).  The spread is the
  regression's residual standard deviation, halved: the residuals are
  right-skewed (a few launches ramp far more than typical), so the full
  standard deviation overstates the band the bulk of launches sit in, and
  half of it scores best on the practice cases while keeping honest
  coverage.  Quantiles are the normal offsets at the five scored levels.

  If too few comparables survive (or the fit is singular) the forecast
  falls back to the plain median ratio curve with its cross-launch spread,
  and if even that is unavailable, to the persistence forecast.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON {"schema":
    "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint",
    "levels", "known", "forecast", "history", "history_hourly", "streams",
    "categories_hourly", "comparables", "notes"}, where `known` holds
    hour 1's print {hour: 1, from, until, viewers, channels, reads} and
    `forecast` lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most 1e9
    (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.
    Anything else is an invalid case, scored as if every quantile equalled
    hour 1's print, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The regression's residual spread is halved (see the module docstring).
SPREAD_SCALE = 0.5
# A comparable counts only if it shows this many release-day hours.
MIN_CURVE_HOURS = 20
# Below this many usable comparables the regression is not trusted.
MIN_COMPARABLES = 8


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def release_hours(comp: dict) -> dict:
    """{hour 1..24 of the comparable's release day: (viewers, channels)}."""
    day = parse_ts(comp["day_start"])
    out = {}
    for pt in comp.get("hourly", []):
        viewers = pt.get("viewers")
        if not viewers:
            continue
        try:
            hour = int((parse_ts(pt["from"]) - day).total_seconds() // 3600) + 1
        except (KeyError, ValueError):
            continue
        if 1 <= hour <= 24:
            out[hour] = (viewers, pt.get("channels") or 0)
    return out


def comparable_curves(case: dict) -> list:
    """Usable comparables, sorted by name for a deterministic fit order.

    Each is (log hour-1 viewers, log hour-1 channels, {hour: log-ratio}).
    """
    curves = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        hours = release_hours(comp)
        first = hours.get(1)
        if not first or first[0] <= 0:
            continue
        v1, ch1 = first
        curve = {}
        for h in range(2, 25):
            v = hours.get(h)
            if v and v[0] > 0:
                curve[h] = math.log(v[0] / v1)
        if len(curve) >= MIN_CURVE_HOURS:
            curves.append((math.log(v1), math.log(max(1, ch1)), curve))
    return curves


def ols(design: list, ys: list):
    """Least squares for design rows [1, x1, x2, ...]; returns (coef, residuals).

    Returns (None, None) when the design is singular.
    """
    k = len(design[0])
    sums = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for row, y in zip(design, ys):
        for i in range(k):
            rhs[i] += row[i] * y
            for j in range(k):
                sums[i][j] += row[i] * row[j]
    mat = [sums[i] + [rhs[i]] for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(mat[r][col]))
        mat[col], mat[piv] = mat[piv], mat[col]
        if abs(mat[col][col]) < 1e-12:
            return None, None
        for r in range(k):
            if r != col:
                f = mat[r][col] / mat[col][col]
                for c in range(col, k + 1):
                    mat[r][c] -= f * mat[col][c]
    coef = [mat[i][k] / mat[i][i] for i in range(k)]
    resid = [y - sum(c * x for c, x in zip(coef, row)) for row, y in zip(design, ys)]
    return coef, resid


def median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def hour_forecast(log_v1: float, log_ch1: float, curves: list, h: int):
    """(median log-ratio, log-spread) for hour h from the comparables."""
    pts = [(lv, lc, c[h]) for lv, lc, c in curves if h in c]
    if len(pts) < 3:
        return 0.0, 0.15
    ys = [p[2] for p in pts]
    if len(pts) >= MIN_COMPARABLES:
        design = [[1.0, p[0], p[1]] for p in pts]
        coef, resid = ols(design, ys)
        if coef is not None:
            mu = coef[0] + coef[1] * log_v1 + coef[2] * log_ch1
            dof = max(1, len(resid) - len(coef))
            sd = math.sqrt(sum(r * r for r in resid) / dof)
            # Guard the extrapolation: a case far bigger than every
            # comparable (GTA VI's hour 1 may be) must not be forecast a
            # ratio outside anything any launch ever showed.  Clamp the
            # predicted log-ratio to the observed range, widened by one
            # residual standard deviation.  Practice cases never bind.
            mu = max(min(ys) - sd, min(max(ys) + sd, mu))
            return mu, SPREAD_SCALE * sd
    # Fallback: the plain median ratio curve and its cross-launch spread.
    mean = sum(ys) / len(ys)
    sd = math.sqrt(sum((y - mean) ** 2 for y in ys) / len(ys))
    return median(ys), SPREAD_SCALE * sd


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_v1 = math.log(p1)
    log_ch1 = math.log(max(1, first.get("channels") or 1))
    curves = comparable_curves(case)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        mu, spread = hour_forecast(log_v1, log_ch1, curves, h)
        q = []
        for z in Z:
            value = p1 * math.exp(mu + z * spread)
            value = min(1e9, max(1.0, value))
            q.append(round2(value))
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
