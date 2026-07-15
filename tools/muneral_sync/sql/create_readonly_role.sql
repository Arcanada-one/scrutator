-- Provision the fixed source-reader login. Supply the password only through
-- the connection-local custom setting before executing this file:
--
--   SELECT set_config('muneral.role_password', '<secret>', false);
--   \ir create_readonly_role.sql
--
-- Do not use psql echo-all/error-verbosity modes while provisioning secrets.
-- The password is never embedded in this repository or returned by the script.
--
-- SECURITY IMPACT: PostgreSQL PUBLIC privileges are additive, so denying TEMP
-- and CREATE to this login requires the database-wide PUBLIC revocations below.
-- They affect every role in the current database. Existing application roles
-- that legitimately create schemas/objects or temporary tables need explicit
-- grants before this script is applied. Ordinary DML roles are unaffected.

DO $provision$
DECLARE
    role_secret text := current_setting('muneral.role_password', true);
BEGIN
    IF role_secret IS NULL OR role_secret = '' THEN
        RAISE EXCEPTION 'muneral.role_password must be set for this connection';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'muneral_kb_reader') THEN
        EXECUTE format(
            'CREATE ROLE muneral_kb_reader LOGIN PASSWORD %L',
            role_secret
        );
    ELSE
        EXECUTE format(
            'ALTER ROLE muneral_kb_reader LOGIN PASSWORD %L',
            role_secret
        );
    END IF;
END;
$provision$;

ALTER ROLE muneral_kb_reader
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
ALTER ROLE muneral_kb_reader SET default_transaction_read_only = on;
ALTER ROLE muneral_kb_reader SET statement_timeout = '15s';

DO $database_privileges$
BEGIN
    EXECUTE format(
        'REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC',
        current_database()
    );
    EXECUTE format(
        'REVOKE CREATE ON DATABASE %I FROM PUBLIC',
        current_database()
    );
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON DATABASE %I FROM muneral_kb_reader',
        current_database()
    );
    EXECUTE format(
        'GRANT CONNECT ON DATABASE %I TO muneral_kb_reader',
        current_database()
    );
END;
$database_privileges$;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM muneral_kb_reader;
GRANT USAGE ON SCHEMA public TO muneral_kb_reader;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM muneral_kb_reader;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM muneral_kb_reader;
REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public FROM muneral_kb_reader;

GRANT SELECT ON TABLE public.users TO muneral_kb_reader;
GRANT SELECT ON TABLE public.workspaces TO muneral_kb_reader;
GRANT SELECT ON TABLE public.projects TO muneral_kb_reader;
GRANT SELECT ON TABLE public.tasks TO muneral_kb_reader;
GRANT SELECT ON TABLE public.task_tags TO muneral_kb_reader;
GRANT SELECT ON TABLE public.task_checklists TO muneral_kb_reader;
GRANT SELECT ON TABLE public.task_dependencies TO muneral_kb_reader;
GRANT SELECT ON TABLE public.task_agents TO muneral_kb_reader;
GRANT SELECT ON TABLE public.agents TO muneral_kb_reader;
GRANT SELECT ON TABLE public.activity_log TO muneral_kb_reader;
GRANT SELECT ON TABLE public.muneral_kb_task_changes TO muneral_kb_reader;

SELECT set_config('muneral.role_password', '', false);
