# Bradestudio-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Bradestudio/Bradestudio-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Bradestudio/Bradestudio-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one.

## Run it

```sh
python3 main.py
```

## How it forecasts

For each trophy it fits a least-squares line of the logit earn rate against log-time since launch over the last 24 hours of reads, shrinking each slope toward the median slope across all trophies and damping the extrapolation toward the target date. The launch night's first refresh dilutes every rate while tracked players flood in, so the full history is U-shaped and a line through all of it points the wrong way; the recent window follows the recovery trend the page is actually on. If the whole history is perfectly flat over at least an hour, the page is treated as frozen and due a refresh, and the median falls by a rarity-dependent dilution instead of one flat shift: the new players a refresh brings hold the easy trophies and almost none of the rare ones. Medians are then made non-increasing along the story chain (chapters 1–4, finale, 100%, platinum), and the five quantiles come from the median logit plus a spread that grows with the horizon and the fit's uncertainty.

---

<sub>Entered on [make.fun](https://make.fun/?ref=qjs4e9sj&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
