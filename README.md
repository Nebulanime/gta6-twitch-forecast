# GTA VI Twitch Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Nebulanime/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Nebulanime/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (2026-11-19 UTC), given only the first hour's
count — built for the make.fun gta6-twitch-ccu-forecast-01 competition.

`main.py` is a self-contained Python 3.13 program (stdlib only, no
network). For every hour of release day it fits, over the case's own
comparables (past big launches' hourly Twitch curves), a regression of
the log viewer ratio `ln(viewers_h / viewers_1)` on `ln(hour-1 viewers)`
and `ln(hour-1 channels)`: launches already hot at midnight UTC ramp
little, quiet ones ramp hard (slope about -0.9 at late hours, R² about
0.88), so the day's peak is nearly scale-free across launches. The
median is the hour-1 print times the predicted ratio; the spread is half
the regression's residual standard deviation, with the predicted ratio
clamped to the range any launch ever showed. Fully deterministic: the
same case always yields the same bytes.

Run it with `python3 -P -s main.py`, pointing `MAKEFUN_INPUT_DIR` at the
case files and `MAKEFUN_OUTPUT_DIR` at where the forecasts go.

## Practice results (local, pinball loss on the log scale)

- p00 Marvel's Wolverine: 0.1465
- p01 Red Dead Redemption 2: 0.1894
- p02 God of War Ragnarök: 0.2024
- p03 Marvel's Spider-Man 2: 0.1849
- Mean: **0.1808** (typical miss 33.6%, 90% coverage 0.978), against the
  house baseline's **0.3871** (typical miss 82.7%, coverage 0.783).

MIT licensed. See `LICENSE`.

<sub>Entered on [make.fun](https://make.fun/?ref=0zsa1rk7&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
