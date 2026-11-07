-- KB-CONS-01-N: native private-generation selection (synthetic-fixture primitive).
--
-- Doc snapshot of the block appended to `schema.sql`. This repository has NO migration runner:
-- `db/connection.py::apply_schema()` executes `schema.sql`, which is the idempotent installer.
-- This numbered file is kept byte-identical to that block for review (a test asserts it).
--
-- Lock order used by every code path (see db/kc_generations.py):
--   1. kc_selections row (scope)  ->  2. kc_generations row  ->  3. member/source/chunk rows.
-- No path acquires a kc_selections lock after a kc_generations lock. Protected writers
-- (stage/seal) enter through generation-first wrappers and never take a selection lock late.
--
-- KC_GENERATION_SCHEMA_BEGIN

-- The namespace/project pair must be enforceable at the database boundary: a unique index over
-- (namespace_id, id) is a valid referenced key and keeps this block purely additive (no table rewrite).
CREATE UNIQUE INDEX IF NOT EXISTS projects_namespace_id_id_key ON projects(namespace_id, id);

CREATE TABLE IF NOT EXISTS kc_corpus_scopes (
    scope_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT NOT NULL REFERENCES namespaces(id) ON DELETE RESTRICT,
    project_id INT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose ~ '^[a-z0-9][a-z0-9._-]{0,63}$'),
    profile TEXT NOT NULL CHECK (profile ~ '^[a-z0-9][a-z0-9._/-]{0,63}$'),
    -- Finite retention/size limits of the registered private profile (a limit refuses staging).
    max_generations INT NOT NULL DEFAULT 8 CHECK (max_generations BETWEEN 1 AND 1000),
    max_members INT NOT NULL DEFAULT 64 CHECK (max_members BETWEEN 1 AND 10000),
    max_manifest_bytes INT NOT NULL DEFAULT 262144 CHECK (max_manifest_bytes BETWEEN 1 AND 16777216),
    max_total_bytes BIGINT NOT NULL DEFAULT 67108864 CHECK (max_total_bytes BETWEEN 1 AND 17179869184),
    max_active_leases INT NOT NULL DEFAULT 32 CHECK (max_active_leases BETWEEN 1 AND 10000),
    lease_ttl_seconds INT NOT NULL DEFAULT 300 CHECK (lease_ttl_seconds BETWEEN 1 AND 86400),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (namespace_id, project_id, purpose),
    UNIQUE (scope_id, namespace_id, project_id),
    FOREIGN KEY (namespace_id, project_id) REFERENCES projects(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS kc_generations (
    scope_id UUID NOT NULL REFERENCES kc_corpus_scopes(scope_id) ON DELETE RESTRICT,
    generation_id TEXT NOT NULL CHECK (generation_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    manifest BYTEA NOT NULL,
    manifest_bytes INT NOT NULL CHECK (manifest_bytes = octet_length(manifest)),
    manifest_sha256 TEXT NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    admission_ref TEXT NOT NULL CHECK (octet_length(admission_ref) BETWEEN 1 AND 256),
    profile_ref TEXT NOT NULL CHECK (octet_length(profile_ref) BETWEEN 1 AND 256),
    state TEXT NOT NULL DEFAULT 'staging'
        CHECK (state IN ('staging', 'sealed', 'rejected', 'gc_pending', 'collected')),
    seal_report JSONB,
    sealed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope_id, generation_id),
    UNIQUE (scope_id, generation_id, manifest_sha256),
    CHECK ((state IN ('sealed', 'gc_pending', 'collected')) = (seal_report IS NOT NULL OR state = 'collected'))
);

CREATE TABLE IF NOT EXISTS kc_generation_members (
    scope_id UUID NOT NULL,
    generation_id TEXT NOT NULL,
    member_id TEXT NOT NULL CHECK (member_id ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}$'),
    semantic_revision TEXT NOT NULL CHECK (octet_length(semantic_revision) BETWEEN 1 AND 128),
    representation_id TEXT NOT NULL CHECK (octet_length(representation_id) BETWEEN 1 AND 128),
    expected_bytes INT NOT NULL CHECK (expected_bytes >= 0),
    expected_sha256 TEXT NOT NULL CHECK (expected_sha256 ~ '^[0-9a-f]{64}$'),
    member_uuid UUID NOT NULL DEFAULT gen_random_uuid(),
    namespace_id INT NOT NULL,
    project_id INT NOT NULL,
    source_path TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    -- Filled exactly once when the member is populated (NULL = not yet populated).
    chunk_set_sha256 TEXT CHECK (chunk_set_sha256 ~ '^[0-9a-f]{64}$'),
    chunk_count INT CHECK (chunk_count >= 1),
    producer TEXT,
    PRIMARY KEY (scope_id, generation_id, member_id),
    UNIQUE (member_uuid),
    UNIQUE (namespace_id, source_path),
    UNIQUE (namespace_id, doc_id),
    FOREIGN KEY (scope_id, generation_id) REFERENCES kc_generations(scope_id, generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (scope_id, namespace_id, project_id)
        REFERENCES kc_corpus_scopes(scope_id, namespace_id, project_id) ON DELETE RESTRICT,
    CHECK (source_path = 'kc-generations/' || scope_id::text || '/' || member_uuid::text),
    CHECK ((chunk_set_sha256 IS NULL) = (chunk_count IS NULL) AND (chunk_set_sha256 IS NULL) = (producer IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_kc_members_generation ON kc_generation_members(scope_id, generation_id);
CREATE INDEX IF NOT EXISTS idx_kc_members_path ON kc_generation_members(namespace_id, project_id, source_path);

CREATE TABLE IF NOT EXISTS kc_selections (
    scope_id UUID PRIMARY KEY REFERENCES kc_corpus_scopes(scope_id) ON DELETE RESTRICT,
    incarnation_id UUID NOT NULL DEFAULT gen_random_uuid(),
    revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0),
    generation_id TEXT,
    manifest_sha256 TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- A default composite FK alone does not check a partially NULL reference.
    CHECK ((generation_id IS NULL) = (manifest_sha256 IS NULL)),
    CHECK ((revision = 0) = (generation_id IS NULL)),
    FOREIGN KEY (scope_id, generation_id, manifest_sha256)
        REFERENCES kc_generations(scope_id, generation_id, manifest_sha256) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS kc_selection_operations (
    scope_id UUID NOT NULL REFERENCES kc_corpus_scopes(scope_id) ON DELETE RESTRICT,
    operation_id TEXT NOT NULL CHECK (operation_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    candidate_generation_id TEXT NOT NULL,
    candidate_manifest_sha256 TEXT NOT NULL,
    expected_incarnation UUID NOT NULL,
    expected_revision BIGINT NOT NULL CHECK (expected_revision >= 0),
    expected_generation_id TEXT,
    committed_incarnation UUID NOT NULL,
    committed_revision BIGINT NOT NULL CHECK (committed_revision >= 1),
    principal_id TEXT NOT NULL,
    authority_ref JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope_id, operation_id),
    UNIQUE (scope_id, committed_incarnation, committed_revision),
    FOREIGN KEY (scope_id, candidate_generation_id, candidate_manifest_sha256)
        REFERENCES kc_generations(scope_id, generation_id, manifest_sha256) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS kc_read_leases (
    lease_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope_id UUID NOT NULL REFERENCES kc_corpus_scopes(scope_id) ON DELETE RESTRICT,
    operation_id TEXT NOT NULL CHECK (operation_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    principal_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    selection_incarnation UUID NOT NULL,
    selection_revision BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'released')),
    authority_ref JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (scope_id, operation_id, principal_id),
    FOREIGN KEY (scope_id, generation_id, manifest_sha256)
        REFERENCES kc_generations(scope_id, generation_id, manifest_sha256) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_kc_leases_generation ON kc_read_leases(scope_id, generation_id, status);

-- ---- immutability / protection triggers (fail-closed backstop; role separation is NOT measured) ----

CREATE OR REPLACE FUNCTION kc_deny_mutation() RETURNS trigger AS $kc$
BEGIN
    RAISE EXCEPTION 'KC_IMMUTABLE: % on % is not permitted', TG_OP, TG_TABLE_NAME USING ERRCODE = 'P0001';
END
$kc$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION kc_generations_guard() RETURNS trigger AS $kc$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: generation rows are tombstoned, never deleted' USING ERRCODE = 'P0001';
    END IF;
    IF NEW.scope_id <> OLD.scope_id OR NEW.generation_id <> OLD.generation_id
       OR NEW.manifest IS DISTINCT FROM OLD.manifest OR NEW.manifest_sha256 <> OLD.manifest_sha256
       OR NEW.manifest_bytes <> OLD.manifest_bytes OR NEW.admission_ref <> OLD.admission_ref
       OR NEW.profile_ref <> OLD.profile_ref THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: generation identity/manifest cannot change' USING ERRCODE = 'P0001';
    END IF;
    IF NEW.state <> OLD.state AND NOT (
           (OLD.state = 'staging' AND NEW.state IN ('sealed', 'rejected'))
        OR (OLD.state = 'sealed' AND NEW.state = 'gc_pending')
        OR (OLD.state = 'gc_pending' AND NEW.state = 'collected')) THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: illegal generation state transition % -> %', OLD.state, NEW.state
            USING ERRCODE = 'P0001';
    END IF;
    IF OLD.state IN ('sealed', 'gc_pending', 'collected') AND NEW.seal_report IS DISTINCT FROM OLD.seal_report THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: seal report is immutable' USING ERRCODE = 'P0001';
    END IF;
    RETURN NEW;
END
$kc$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION kc_members_guard() RETURNS trigger AS $kc$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: member rows are retained' USING ERRCODE = 'P0001';
    END IF;
    IF NEW.scope_id <> OLD.scope_id OR NEW.generation_id <> OLD.generation_id OR NEW.member_id <> OLD.member_id
       OR NEW.semantic_revision <> OLD.semantic_revision OR NEW.representation_id <> OLD.representation_id
       OR NEW.expected_bytes <> OLD.expected_bytes OR NEW.expected_sha256 <> OLD.expected_sha256
       OR NEW.member_uuid <> OLD.member_uuid OR NEW.namespace_id <> OLD.namespace_id
       OR NEW.project_id <> OLD.project_id OR NEW.source_path <> OLD.source_path OR NEW.doc_id <> OLD.doc_id THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: member identity cannot change' USING ERRCODE = 'P0001';
    END IF;
    IF OLD.chunk_set_sha256 IS NOT NULL AND (NEW.chunk_set_sha256 IS DISTINCT FROM OLD.chunk_set_sha256
       OR NEW.chunk_count IS DISTINCT FROM OLD.chunk_count OR NEW.producer IS DISTINCT FROM OLD.producer) THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: a populated member cannot be replaced' USING ERRCODE = 'P0001';
    END IF;
    RETURN NEW;
END
$kc$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION kc_selections_guard() RETURNS trigger AS $kc$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: a selection row is never deleted' USING ERRCODE = 'P0001';
    END IF;
    IF NEW.scope_id <> OLD.scope_id OR NEW.incarnation_id <> OLD.incarnation_id THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: selection incarnation cannot be overwritten' USING ERRCODE = 'P0001';
    END IF;
    IF NEW.revision <> OLD.revision + 1 THEN
        RAISE EXCEPTION 'KC_IMMUTABLE: selection revision must advance by exactly one (no reset, no skip)'
            USING ERRCODE = 'P0001';
    END IF;
    RETURN NEW;
END
$kc$ LANGUAGE plpgsql;

-- Every committed selection change must carry its operation record from the SAME transaction.
CREATE OR REPLACE FUNCTION kc_selections_require_operation() RETURNS trigger AS $kc$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM kc_selection_operations o
        WHERE o.scope_id = NEW.scope_id AND o.committed_incarnation = NEW.incarnation_id
          AND o.committed_revision = NEW.revision AND o.candidate_generation_id = NEW.generation_id
          AND o.candidate_manifest_sha256 = NEW.manifest_sha256
          AND o.xmin = pg_current_xact_id()::xid) THEN
        RAISE EXCEPTION 'KC_SELECTION_OPERATION_MISSING: selection revision % has no operation row in this transaction',
            NEW.revision USING ERRCODE = 'P0001';
    END IF;
    RETURN NULL;
END
$kc$ LANGUAGE plpgsql;

-- Reserved private storage prefix: chunks / exact bodies / vectors of member paths.
CREATE OR REPLACE FUNCTION kc_private_path_guard() RETURNS trigger AS $kc$
DECLARE
    v_ns INT;
    v_path TEXT;
    v_state TEXT;
    v_scope UUID;
    v_gen TEXT;
BEGIN
    IF TG_TABLE_NAME = 'sparse_vectors' THEN
        SELECT c.namespace_id, c.source_path INTO v_ns, v_path FROM chunks c
         WHERE c.id = CASE WHEN TG_OP = 'DELETE' THEN OLD.chunk_id ELSE NEW.chunk_id END;
        IF v_path IS NULL OR v_path NOT LIKE 'kc-generations/%' THEN
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END IF;
    ELSE
        v_ns := CASE WHEN TG_OP = 'DELETE' THEN OLD.namespace_id ELSE NEW.namespace_id END;
        v_path := CASE WHEN TG_OP = 'DELETE' THEN OLD.source_path ELSE NEW.source_path END;
    END IF;
    SELECT m.scope_id, m.generation_id, g.state INTO v_scope, v_gen, v_state
      FROM kc_generation_members m JOIN kc_generations g USING (scope_id, generation_id)
     WHERE m.namespace_id = v_ns AND m.source_path = v_path;
    IF v_state IS NULL THEN
        RAISE EXCEPTION 'KC_PRIVATE_PATH: reserved path has no registered member' USING ERRCODE = 'P0001';
    END IF;
    IF TG_OP = 'INSERT' AND v_state = 'staging'
       AND current_setting('scrutator.kc_stage_writer', true) = v_scope::text || '/' || v_gen THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'KC_IMMUTABLE: % on private generation path (state=%) is not permitted', TG_OP, v_state
        USING ERRCODE = 'P0001';
END
$kc$ LANGUAGE plpgsql;

DO $kc$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'kc_scopes_immutable') THEN
        CREATE TRIGGER kc_scopes_immutable BEFORE UPDATE OR DELETE ON kc_corpus_scopes
            FOR EACH ROW EXECUTE FUNCTION kc_deny_mutation();
        CREATE TRIGGER kc_operations_immutable BEFORE UPDATE OR DELETE ON kc_selection_operations
            FOR EACH ROW EXECUTE FUNCTION kc_deny_mutation();
        CREATE TRIGGER kc_generations_guard_trg BEFORE UPDATE OR DELETE ON kc_generations
            FOR EACH ROW EXECUTE FUNCTION kc_generations_guard();
        CREATE TRIGGER kc_members_guard_trg BEFORE UPDATE OR DELETE ON kc_generation_members
            FOR EACH ROW EXECUTE FUNCTION kc_members_guard();
        CREATE TRIGGER kc_selections_guard_trg BEFORE UPDATE OR DELETE ON kc_selections
            FOR EACH ROW EXECUTE FUNCTION kc_selections_guard();
        CREATE CONSTRAINT TRIGGER kc_selections_operation_trg AFTER UPDATE ON kc_selections
            DEFERRABLE INITIALLY DEFERRED
            FOR EACH ROW EXECUTE FUNCTION kc_selections_require_operation();
        CREATE TRIGGER kc_chunks_private_ins BEFORE INSERT ON chunks
            FOR EACH ROW WHEN (NEW.source_path LIKE 'kc-generations/%')
            EXECUTE FUNCTION kc_private_path_guard();
        CREATE TRIGGER kc_chunks_private_upd BEFORE UPDATE ON chunks
            FOR EACH ROW WHEN (OLD.source_path LIKE 'kc-generations/%' OR NEW.source_path LIKE 'kc-generations/%')
            EXECUTE FUNCTION kc_private_path_guard();
        CREATE TRIGGER kc_chunks_private_del BEFORE DELETE ON chunks
            FOR EACH ROW WHEN (OLD.source_path LIKE 'kc-generations/%')
            EXECUTE FUNCTION kc_private_path_guard();
        CREATE TRIGGER kc_evidence_private_ins BEFORE INSERT ON evidence_documents
            FOR EACH ROW WHEN (NEW.source_path LIKE 'kc-generations/%')
            EXECUTE FUNCTION kc_private_path_guard();
        CREATE TRIGGER kc_evidence_private_upd BEFORE UPDATE ON evidence_documents
            FOR EACH ROW WHEN (OLD.source_path LIKE 'kc-generations/%' OR NEW.source_path LIKE 'kc-generations/%')
            EXECUTE FUNCTION kc_private_path_guard();
        CREATE TRIGGER kc_evidence_private_del BEFORE DELETE ON evidence_documents
            FOR EACH ROW WHEN (OLD.source_path LIKE 'kc-generations/%')
            EXECUTE FUNCTION kc_private_path_guard();
        -- sparse_vectors carries no path: the guard resolves it through the chunk row (PK lookup).
        CREATE TRIGGER kc_sparse_private_guard BEFORE INSERT OR UPDATE OR DELETE ON sparse_vectors
            FOR EACH ROW EXECUTE FUNCTION kc_private_path_guard();
    END IF;
END
$kc$;

-- KC_GENERATION_SCHEMA_END
