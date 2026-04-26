-- LTM-0012: Temporal layer (T in TEMPR)
-- Adds entity_events table + GiST range index for time-aware recall.
-- Idempotent — safe to re-apply.

CREATE EXTENSION IF NOT EXISTS btree_gist;

CREATE TABLE IF NOT EXISTS entity_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT REFERENCES namespaces(id) NOT NULL,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    when_t TIMESTAMPTZ,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    description TEXT,
    properties JSONB DEFAULT '{}',
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    superseded_by UUID REFERENCES entity_events(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (namespace_id, entity_id, event_type, when_t)
);

CREATE INDEX IF NOT EXISTS idx_events_entity ON entity_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_events_when ON entity_events(when_t);
CREATE INDEX IF NOT EXISTS idx_events_namespace ON entity_events(namespace_id);
CREATE INDEX IF NOT EXISTS idx_events_valid_period
    ON entity_events USING gist (tstzrange(valid_from, valid_to, '[)'));
CREATE INDEX IF NOT EXISTS idx_events_active
    ON entity_events(entity_id) WHERE valid_to IS NULL;
