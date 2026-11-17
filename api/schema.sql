-- The table that logs every prediction the live app makes.
-- Run this once in Neon's SQL Editor. api/store.py reuses it for its test tables.

CREATE TABLE IF NOT EXISTS predictions (
    id          BIGSERIAL PRIMARY KEY,               -- 1, 2, 3... added automatically
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),  -- when it was made, with time zone
    track_genre TEXT NOT NULL,                       -- copied out so it's easy to group by
    features    JSONB NOT NULL,                      -- the whole song the visitor sent
    prediction  REAL NOT NULL                        -- the score the model gave
);
