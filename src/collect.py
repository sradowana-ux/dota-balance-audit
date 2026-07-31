"""Collect public Dota 2 matches from the OpenDota API.

Pages backwards through /api/publicMatches using the `less_than_match_id`
cursor, applies quality filters, and writes a CSV.

Quality filters, and why each one is here:

  * Both teams must contain five distinct, non-zero hero IDs. OpenDota returns
    zero-filled hero arrays for very recent matches that its parser has not
    reached yet; those rows carry no draft information.
  * duration >= 900 seconds. Dota allows a team to abandon early, which ends
    the match with an outcome that reflects a disconnect rather than the draft.
    Fifteen minutes is the conventional cut-off.
  * avg_rank_tier must be present, so every row can be bucketed by skill.

Usage:
    python src/collect.py --target 40000 --out data/matches.csv

The public API is rate-limited to 60 calls/minute and 2000 calls/day without a
key, so the default delay is 1 second between calls. Roughly 100 matches
arrive per call, of which ~55% survive filtering.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import requests

API = "https://api.opendota.com/api/publicMatches"

FIELDS = [
    "match_id",
    "start_time",
    "duration",
    "lobby_type",
    "game_mode",
    "avg_rank_tier",
    "radiant_win",
    "radiant_team",
    "dire_team",
]


def is_valid(match: dict) -> bool:
    radiant = match.get("radiant_team")
    dire = match.get("dire_team")
    if not isinstance(radiant, list) or not isinstance(dire, list):
        return False
    if len(radiant) != 5 or len(dire) != 5:
        return False
    if any(h in (None, 0) for h in radiant + dire):
        return False
    if len(set(radiant + dire)) != 10:
        return False
    if (match.get("duration") or 0) < 900:
        return False
    if match.get("avg_rank_tier") is None:
        return False
    return True


def collect(target: int, start_cursor: int | None, delay: float, max_calls: int):
    rows: list[dict] = []
    cursor = start_cursor
    calls = 0
    session = requests.Session()

    while len(rows) < target and calls < max_calls:
        params = {} if cursor is None else {"less_than_match_id": cursor}
        try:
            response = session.get(API, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"  request failed ({exc}); backing off", file=sys.stderr)
            time.sleep(5)
            continue

        calls += 1
        if response.status_code == 429:
            print("  rate limited; sleeping 60s", file=sys.stderr)
            time.sleep(60)
            continue
        if not response.ok:
            time.sleep(5)
            continue

        batch = response.json()
        if not batch:
            break

        for match in batch:
            if not is_valid(match):
                continue
            rows.append(
                {
                    "match_id": match["match_id"],
                    "start_time": match["start_time"],
                    "duration": match["duration"],
                    "lobby_type": match["lobby_type"],
                    "game_mode": match["game_mode"],
                    "avg_rank_tier": match["avg_rank_tier"],
                    "radiant_win": int(bool(match["radiant_win"])),
                    "radiant_team": " ".join(str(h) for h in match["radiant_team"]),
                    "dire_team": " ".join(str(h) for h in match["dire_team"]),
                }
            )

        cursor = batch[-1]["match_id"]
        print(f"  call {calls}: {len(rows)} matches kept", file=sys.stderr)
        time.sleep(delay)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=40000)
    parser.add_argument("--out", default="data/matches.csv")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=1500)
    parser.add_argument(
        "--start-cursor",
        type=int,
        default=None,
        help="Page backwards from this match_id. Omit to start from the most "
        "recent matches (note: the newest few thousand are usually unparsed).",
    )
    args = parser.parse_args()

    rows = collect(args.target, args.start_cursor, args.delay, args.max_calls)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} matches to {args.out}")


if __name__ == "__main__":
    main()
