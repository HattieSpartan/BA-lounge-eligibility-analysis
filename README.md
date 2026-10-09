# Lounge Eligibility Lookup for Heathrow Terminal 3

A data analysis project built from the British Airways Data Science job simulation on Forage (Task 1). The simulation is a learning exercise, not paid or client work for British Airways, and this repository is not affiliated with or endorsed by British Airways.

## The question

British Airways wants to estimate how many passengers on a future flight schedule will be eligible for each of three lounge tiers at Heathrow Terminal 3. Given a flight schedule with seats but no passenger counts, what is the simplest lookup table that gives reliable estimates, and how do we know it can be trusted?

## Result

Group flights by haul type only. Each rate is eligible passengers as a share of seats.

| Haul type | Tier 1 (hypothetical) | Tier 2 | Tier 3 |
|---|---|---|---|
| Short-haul (Europe) | 0.34% | 4.40% | 16.85% |
| Long-haul (North America, Middle East, Asia) | 0.20% | 2.74% | 10.47% |

Tier 1 is the Concorde Room, which Terminal 3 does not have today, so Tier 1 figures show how many passengers would qualify if one existed.

To apply it: multiply a flight's seats by the rate for its haul type, then add up across flights.

## What was tested

**Which grouping?** I compared haul type, destination region and time of day. Region adds nothing beyond haul (every short-haul flight goes to Europe, and the three long-haul regions are within about 0.3 percentage points). Time of day moves rates by under 1 percentage point and is not statistically reliable.

**Is it stable?** Monthly Tier 3 rates stay within about 0.9 percentage points of each other from April to October 2025.

**Does it predict unseen flights?** Rates were built on April to June (4,407 flights) and tested on July (1,389 flights):

| | Estimated | Actual | Error |
|---|---|---|---|
| Tier 1 | 853 | 825 | +3.4% |
| Tier 2 | 11,066 | 10,868 | +1.8% |
| Tier 3 | 42,295 | 41,721 | +1.4% |

Totals are close, but single flights are not: the Tier 3 estimate is off by about 12.6 passengers per flight on average (roughly 42% of a typical flight's count), and daily Tier 3 totals are off by 4.4% on average (worst day 14.3%). **Use the table for totals across many flights, not for predicting one flight.**

**Would adding time of day help?** A haul-plus-time-of-day lookup was compared with the haul-only lookup on the same July flights. Tier 3 flight-level error was 12.57 against 12.58 passengers, and the gain was not distinguishable from zero. It is not worth the extra rows.

## A lesson from the statistics

My first test of time of day used a binomial model that treats every seat as an independent trial. It produced p-values below 0.0001 for some tiers. An audit showed those p-values were too small: eligible counts vary 2 to 3.5 times more from flight to flight than independent seats would allow (dispersion between 1.7 and 11.5). Allowing for that extra variation (a quasi-binomial test, cross-checked with a permutation test and a Holm correction across six tests) removed every significant result. The p-values from the invalid model would have justified extra rows in the table that the held-out data shows add nothing. See `audit/`.

## Assumptions and limitations

- The data contains seats, not passengers, so every rate is a share of seats and does not account for load factor.
- The tier counts are not nested (Tier 1 exceeds Tier 2 on 459 flights and Tier 3 on 126), but the data does not show whether passengers are counted in more than one tier. Tiers are therefore reported separately and are never added together.
- Rates assume the future fleet, route and loyalty mix looks like April to October 2025. Refresh them when it changes (Tier 3 is about 6% on the A380 and about 14% on the B787 in this data).
- Connecting passengers and time spent in the lounge are not modelled, so this is an input to planning, not a measure of lounge occupancy.
- A non-significant time-of-day result does not prove time of day never matters. It means any effect is too small to detect reliably here (largest observed Tier 3 difference was about 0.4 percentage points).

## Repository

```
lounge_eligibility_model.py     Reusable script: lookup table, monthly stability, holdout test, time-of-day tests
notebooks/                      Step-by-step analysis notebook with commentary
audit/                          Statistical audit of the time-of-day tests (script, results, chart)
results/                        Aggregated output tables (no flight-level data)
```

## How to run

```
pip install -r requirements.txt
python lounge_eligibility_model.py path/to/schedule.xlsx --out-dir results
python audit/lounge_time_of_day_audit.py path/to/schedule.xlsx --out-dir audit
```

## Data

The flight schedule dataset belongs to the Forage / British Airways job simulation and is not included here. Download it from the simulation to reproduce the results. The `results/` folder contains only aggregated tables.

## Author

Harriet Joseph. Data analytics, research and AI. [LinkedIn](https://linkedin.com/in/harriet-joseph-5669b921b) | [Portfolio](https://datascienceportfol.io/harrietjoseph)
