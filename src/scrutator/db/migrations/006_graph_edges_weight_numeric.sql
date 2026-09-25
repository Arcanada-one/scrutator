-- A2-308 — graph_edges.weight: REAL -> NUMERIC.
--
-- `weight` carries two different kinds of number: dreamer similarity scores (fractions) and
-- the Part 5 §5.2 cost quantities, which are integers (tokens, bytes, wall_time_s,
-- usd_micros). REAL is float32, so it stores integers exactly only up to 2^24 = 16777216.
-- A2-305 measured live weights up to 9 600 000 — a factor of 1.7 below the point where
-- consecutive integers stop being representable and a value is silently rounded to its
-- neighbour. BIGINT would fix the integers and destroy the fractions; NUMERIC is exact for
-- both, which is why it is the target here.
--
-- The cast goes through text on purpose. `weight::numeric` would carry the float32
-- artefact into the new column (0.85 REAL becomes 0.850000023841858); `weight::text`
-- renders the shortest decimal that round-trips to the same float32, so an 0.85 that was
-- written as 0.85 comes back as 0.85. Precision already lost by float32 cannot be
-- recovered by any cast — this only avoids making it permanent and uglier.

ALTER TABLE graph_edges
    ALTER COLUMN weight TYPE NUMERIC USING weight::text::numeric,
    ALTER COLUMN weight SET DEFAULT 1.0;
