-- Hero-level win and pick rates, overall and by skill bracket.
--
-- These are the raw inputs to the balance tests. No significance is decided
-- here on purpose: this layer produces counts, and the statistics layer decides
-- what the counts mean. Mixing the two is how people end up eyeballing a
-- win-rate column and calling heroes broken.

-- Overall, one row per hero.
CREATE OR REPLACE TABLE hero_winrates AS
SELECT
    hm.hero_id,
    h.name,
    h.primary_attr,
    COUNT(*)                                          AS games,
    SUM(hm.won)                                       AS wins,
    SUM(hm.won) / CAST(COUNT(*) AS DOUBLE)            AS win_rate,
    -- pick rate = share of matches this hero appeared in. Denominator is
    -- matches, not hero-rows, so a hero in every game would score 1.0.
    COUNT(*) / CAST((SELECT COUNT(*) FROM matches) AS DOUBLE) AS pick_rate
FROM hero_matches hm
JOIN heroes h USING (hero_id)
GROUP BY hm.hero_id, h.name, h.primary_attr
ORDER BY win_rate DESC;

-- Same, split by skill bracket. Balance is not a single number: a hero can be
-- oppressive in low brackets and unplayable in high ones, and a global average
-- hides exactly that.
CREATE OR REPLACE TABLE hero_winrates_by_bracket AS
SELECT
    hm.bracket,
    hm.hero_id,
    h.name,
    COUNT(*)                               AS games,
    SUM(hm.won)                            AS wins,
    SUM(hm.won) / CAST(COUNT(*) AS DOUBLE) AS win_rate
FROM hero_matches hm
JOIN heroes h USING (hero_id)
GROUP BY hm.bracket, hm.hero_id, h.name
HAVING COUNT(*) >= 100
ORDER BY hm.bracket, win_rate DESC;

-- Side balance. Radiant advantage is a structural property of the map, so it
-- should be measured on matches, not hero-rows.
CREATE OR REPLACE TABLE side_balance AS
SELECT
    COUNT(*)                                      AS matches,
    SUM(radiant_win)                              AS radiant_wins,
    SUM(radiant_win) / CAST(COUNT(*) AS DOUBLE)   AS radiant_win_rate
FROM matches;

CREATE OR REPLACE TABLE side_balance_by_bracket AS
SELECT
    bracket,
    COUNT(*)                                    AS matches,
    SUM(radiant_win)                            AS radiant_wins,
    SUM(radiant_win) / CAST(COUNT(*) AS DOUBLE) AS radiant_win_rate
FROM matches
GROUP BY bracket
ORDER BY bracket;

-- How far apart are the extremes? A one-line summary of "how balanced is the
-- roster", restricted to heroes with enough games for the rate to mean anything.
CREATE OR REPLACE TABLE roster_spread AS
SELECT
    COUNT(*)                        AS heroes,
    MIN(win_rate)                   AS min_win_rate,
    MAX(win_rate)                   AS max_win_rate,
    MAX(win_rate) - MIN(win_rate)   AS spread,
    STDDEV_SAMP(win_rate)           AS sd
FROM hero_winrates
WHERE games >= 500;
