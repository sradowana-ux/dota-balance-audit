-- Build the analysis tables from the raw match export.
--
-- The raw file stores a match as one row with two space-separated hero lists.
-- Almost every balance question is asked at the hero-match grain instead
-- ("how did hero H do, in which side, at which bracket"), so the first job is
-- to unpivot ten heroes per match into ten rows.
--
-- Skill bracket: OpenDota's avg_rank_tier packs medal and star into one number
-- (54 = Legend 4). Integer division by 10 recovers the medal, which is the
-- grain balance is actually discussed at.

CREATE OR REPLACE TABLE matches AS
SELECT
    match_id,
    to_timestamp(start_time)                     AS started_at,
    duration,
    lobby_type,
    game_mode,
    avg_rank_tier,
    avg_rank_tier // 10                          AS bracket,
    radiant_win,
    radiant_team,
    dire_team
FROM read_csv_auto(getvariable('matches_path'), header = true)
WHERE game_mode = 22;          -- All Draft only; Turbo is a different game

-- One row per hero per match. `won` is resolved relative to the side the hero
-- was on, so downstream queries never have to re-derive it.
CREATE OR REPLACE TABLE hero_matches AS
WITH sides AS (
    SELECT match_id, bracket, duration, started_at,
           'radiant' AS side, radiant_team AS team, radiant_win AS side_won
    FROM matches
    UNION ALL
    SELECT match_id, bracket, duration, started_at,
           'dire'    AS side, dire_team    AS team, 1 - radiant_win AS side_won
    FROM matches
)
SELECT
    s.match_id,
    s.bracket,
    s.side,
    CAST(hero AS INTEGER) AS hero_id,
    s.side_won            AS won
FROM sides s,
     UNNEST(str_split(s.team, ' ')) AS t(hero)
WHERE hero <> '';

CREATE OR REPLACE TABLE heroes AS
SELECT
    hero_id,
    name,
    primary_attr,
    attack_type,
    roles
FROM read_csv_auto(getvariable('heroes_path'), header = true);

-- Sanity: every match must contribute exactly ten hero-rows. If this returns
-- anything, the unpivot is wrong and every number downstream is wrong with it.
CREATE OR REPLACE TABLE integrity_check AS
SELECT match_id, COUNT(*) AS hero_rows
FROM hero_matches
GROUP BY match_id
HAVING COUNT(*) <> 10;
