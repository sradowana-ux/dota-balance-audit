"""Tests for the draft encoding.

The property that matters most is antisymmetry: mirroring a draft must negate
the feature vector exactly. If that breaks, the model silently stops being
side-symmetric and the interpretation of the intercept as "Radiant map
advantage" stops being true -- which no accuracy metric would catch.

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import (  # noqa: E402
    build_hero_index,
    build_pair_index,
    encode_drafts,
    encode_drafts_pairwise,
    encode_single_draft,
    parse_team,
)

HERO_IDS = [1, 2, 5, 8, 14, 20, 26, 35, 42, 74, 84, 101]


@pytest.fixture
def hero_index():
    return build_hero_index(HERO_IDS)


@pytest.fixture
def drafts():
    return pd.DataFrame(
        {
            "radiant_team": ["1 2 5 8 14", "20 26 35 42 74"],
            "dire_team": ["20 26 35 42 74", "1 2 5 8 84"],
            "radiant_win": [1, 0],
        }
    )


def test_parse_team():
    assert parse_team("14 42 18 9 11") == [14, 42, 18, 9, 11]


def test_hero_index_is_contiguous(hero_index):
    assert sorted(hero_index.values()) == list(range(len(HERO_IDS)))


def test_encoding_has_five_per_side(hero_index, drafts):
    X = encode_drafts(drafts, hero_index)
    for row in X:
        assert (row == 1).sum() == 5
        assert (row == -1).sum() == 5
        assert row.sum() == 0


def test_encoding_is_antisymmetric(hero_index, drafts):
    """Swapping the teams must negate the feature vector exactly."""
    mirrored = drafts.rename(
        columns={"radiant_team": "dire_team", "dire_team": "radiant_team"}
    )
    np.testing.assert_array_equal(
        encode_drafts(drafts, hero_index), -encode_drafts(mirrored, hero_index)
    )


def test_pairwise_encoding_is_antisymmetric(hero_index, drafts):
    pair_index = build_pair_index(len(hero_index))
    mirrored = drafts.rename(
        columns={"radiant_team": "dire_team", "dire_team": "radiant_team"}
    )
    original = encode_drafts_pairwise(drafts, hero_index, pair_index).toarray()
    flipped = encode_drafts_pairwise(mirrored, hero_index, pair_index).toarray()
    np.testing.assert_allclose(original, -flipped)


def test_pairwise_block_sizes(hero_index, drafts):
    n = len(hero_index)
    pair_index = build_pair_index(n)
    X = encode_drafts_pairwise(drafts, hero_index, pair_index)
    assert X.shape == (len(drafts), n + 2 * len(pair_index))
    # 10 hero terms + 2 * C(5,2) synergy terms + 25 counter terms per match
    assert X.getnnz(axis=1).tolist() == [10 + 20 + 25] * len(drafts)


def test_single_draft_matches_batch(hero_index, drafts):
    batch = encode_drafts(drafts, hero_index)
    single = encode_single_draft(
        parse_team(drafts.loc[0, "radiant_team"]),
        parse_team(drafts.loc[0, "dire_team"]),
        hero_index,
    )
    np.testing.assert_array_equal(batch[0:1], single)


def test_unknown_hero_is_ignored_not_crashed(hero_index):
    """A hero added in a patch newer than the training data must not raise."""
    x = encode_single_draft([1, 2, 5, 8, 999], [20, 26, 35, 42, 74], hero_index)
    assert (x == 1).sum() == 4
    assert (x == -1).sum() == 5
