"""Feature encoding for Dota 2 draft win prediction.

The core design choice here is the *antisymmetric* (signed) hero encoding.

A draft is two sets of five heroes. The naive encoding gives each hero two
columns -- one for "on Radiant", one for "on Dire" -- producing 2 * n_heroes
features. That representation lets the model learn two independent weights for
the same hero, which is wrong: Dota is (approximately) side-symmetric, so a
hero that is worth +w to Radiant should be worth -w to Dire.

We instead use a single column per hero:

    x[h] = +1  if hero h is on Radiant
           -1  if hero h is on Dire
            0  otherwise

This halves the parameter count and hard-codes the constraint that swapping the
two teams flips the sign of the logit. For a linear model that means
P(radiant wins | draft) = 1 - P(radiant wins | mirrored draft) exactly, which
is a property you want and would otherwise only get approximately.

The trade-off, stated plainly: this encoding cannot represent Radiant-side map
advantage as a hero-specific effect. That effect is absorbed into the intercept
instead, which is the right place for it given that side advantage in Dota is
overwhelmingly a map property rather than a hero property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hero IDs in Dota 2 are not contiguous (there are gaps where heroes were
# renumbered), so we build an explicit id -> column index map rather than
# assuming index == id.


def build_hero_index(hero_ids) -> dict[int, int]:
    """Map raw Dota hero IDs onto contiguous column indices."""
    return {int(h): i for i, h in enumerate(sorted(set(int(x) for x in hero_ids)))}


def parse_team(cell: str) -> list[int]:
    """'14 42 18 9 11' -> [14, 42, 18, 9, 11]"""
    return [int(x) for x in str(cell).split()]


def encode_drafts(df: pd.DataFrame, hero_index: dict[int, int]) -> np.ndarray:
    """Encode a match dataframe into the signed hero matrix described above.

    Returns an (n_matches, n_heroes) float32 array with exactly five +1 entries
    and five -1 entries per row.
    """
    n_heroes = len(hero_index)
    X = np.zeros((len(df), n_heroes), dtype=np.float32)

    for row, (radiant, dire) in enumerate(
        zip(df["radiant_team"].to_numpy(), df["dire_team"].to_numpy())
    ):
        for hero in parse_team(radiant):
            col = hero_index.get(hero)
            if col is not None:
                X[row, col] = 1.0
        for hero in parse_team(dire):
            col = hero_index.get(hero)
            if col is not None:
                X[row, col] = -1.0

    return X


def encode_single_draft(
    radiant: list[int], dire: list[int], hero_index: dict[int, int]
) -> np.ndarray:
    """Encode one draft for inference. Shape (1, n_heroes)."""
    x = np.zeros((1, len(hero_index)), dtype=np.float32)
    for hero in radiant:
        col = hero_index.get(int(hero))
        if col is not None:
            x[0, col] = 1.0
    for hero in dire:
        col = hero_index.get(int(hero))
        if col is not None:
            x[0, col] = -1.0
    return x


def build_pair_index(n_heroes: int) -> dict[tuple[int, int], int]:
    """Index every unordered hero pair (i < j) onto a column offset."""
    index = {}
    k = 0
    for i in range(n_heroes):
        for j in range(i + 1, n_heroes):
            index[(i, j)] = k
            k += 1
    return index


def encode_drafts_pairwise(
    df: pd.DataFrame, hero_index: dict[int, int], pair_index: dict[tuple[int, int], int]
):
    """Signed hero features plus synergy and counter pair features.

    Three blocks, all antisymmetric under swapping the two teams:

      1. hero      (n)      as in `encode_drafts`
      2. synergy   (n*(n-1)/2)  +1 if the pair is together on Radiant,
                                -1 if together on Dire, 0 otherwise
      3. counter   (n*(n-1)/2)  +1 if the lower-indexed hero of the pair is on
                                Radiant and the higher on Dire, -1 for the
                                reverse

    This is ~16k columns for 127 heroes, which is more parameters than we have
    matches. It only works under heavy L2 regularisation, and whether it earns
    its keep is an empirical question the evaluation answers rather than
    assumes.

    Returns a scipy CSR matrix.
    """
    from scipy import sparse

    n = len(hero_index)
    n_pairs = len(pair_index)

    rows, cols, vals = [], [], []
    for row, (radiant, dire) in enumerate(
        zip(df["radiant_team"].to_numpy(), df["dire_team"].to_numpy())
    ):
        R = [hero_index[h] for h in parse_team(radiant) if h in hero_index]
        D = [hero_index[h] for h in parse_team(dire) if h in hero_index]

        for col in R:
            rows.append(row); cols.append(col); vals.append(1.0)
        for col in D:
            rows.append(row); cols.append(col); vals.append(-1.0)

        for team, sign in ((R, 1.0), (D, -1.0)):
            for a in range(len(team)):
                for b in range(a + 1, len(team)):
                    i, j = sorted((team[a], team[b]))
                    if i == j:
                        continue
                    rows.append(row); cols.append(n + pair_index[(i, j)]); vals.append(sign)

        for a in R:
            for b in D:
                if a == b:
                    continue
                i, j = (a, b) if a < b else (b, a)
                sign = 1.0 if a < b else -1.0
                rows.append(row); cols.append(n + n_pairs + pair_index[(i, j)]); vals.append(sign)

    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(df), n + 2 * n_pairs), dtype=np.float32
    )


def load_matches(path: str, game_mode: int | None = 22) -> pd.DataFrame:
    """Load the collected CSV and apply mode filtering.

    game_mode 22 is All Draft, the standard ranked matchmaking mode. Turbo
    (mode 23) is excluded by default: it has roughly half the game duration and
    a substantially different economy, so hero win rates there are not drawn
    from the same distribution. Mixing the two would mean fitting one set of
    hero weights to two different games.
    """
    df = pd.read_csv(path)
    if game_mode is not None:
        df = df[df["game_mode"] == game_mode]
    df = df.drop_duplicates(subset="match_id")
    # Chronological order: match_id is monotonically increasing with time.
    df = df.sort_values("match_id").reset_index(drop=True)
    return df
