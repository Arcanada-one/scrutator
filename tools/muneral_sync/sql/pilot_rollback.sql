-- Remove only the deterministic LTM-0025 pilot rows. Serializable predicate
-- checks make a collision or later foreign child abort before any DELETE.
-- muneral_kb_task_changes is intentionally retained as a deleted tombstone.
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

DO $rollback_preflight$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM users
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND name = 'LTM-0025 Pilot Agent'
          AND github_id IS NULL AND telegram_id IS NULL AND avatar_url IS NULL
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
          AND updated_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR NOT EXISTS (
        SELECT 1 FROM workspaces
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
          AND slug = 'ltm-0025-pilot' AND name = 'LTM-0025 Pilot'
          AND owner_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND subscription_tier = 'free'
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR NOT EXISTS (
        SELECT 1 FROM projects
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
          AND workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
          AND slug = 'long-term-memory' AND name = 'Long Term Memory'
          AND description = 'Deterministic Muneral to KB graph-merge pilot project.'
          AND repo_url IS NULL
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR NOT EXISTS (
        SELECT 1 FROM tasks
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND project_id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
          AND sprint_id IS NULL AND parent_id IS NULL
          AND title = 'LTM-0025 Muneral to KB graph-merge pilot'
          AND description = 'Prove deterministic task graph ingestion, recall, and idempotent replay.'
          AND status = 'in_progress' AND priority = 'high' AND due_date IS NULL
          AND estimate_hours = 2.00
          AND created_by_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND actor_type = 'agent'
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
          AND updated_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR (
        SELECT count(*) FROM task_tags
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND tag IN ('graph-merge', 'pilot')
    ) <> 2 OR NOT EXISTS (
        SELECT 1 FROM task_checklists
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000005'
          AND task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND text = 'Pilot task graph is proven in the KB through the granted principal'
          AND checked = false AND position = 1
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR NOT EXISTS (
        SELECT 1 FROM activity_log
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000006'
          AND task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
          AND actor_type = 'agent'
          AND actor_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND action = 'task.created'
          AND payload = '{"source":"LTM-0025","mode":"pilot"}'::jsonb
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) THEN
        RAISE EXCEPTION 'pilot rollback identity mismatch';
    END IF;

    IF EXISTS (
        SELECT 1 FROM task_dependencies
        WHERE from_task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
           OR to_task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM task_agents WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM task_tags
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND tag NOT IN ('graph-merge', 'pilot')
    ) OR EXISTS (
        SELECT 1 FROM task_checklists
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND id <> '7d2c0e8a-4b7e-4c01-8d33-100000000005'
    ) OR EXISTS (
        SELECT 1 FROM task_git_refs WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM task_field_state WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM agent_field_reads WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM tasks WHERE parent_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM tasks
        WHERE project_id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
          AND id <> '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM milestones WHERE project_id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
    ) OR EXISTS (
        SELECT 1 FROM sprints WHERE project_id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
    ) OR EXISTS (
        SELECT 1 FROM projects
        WHERE workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
          AND id <> '7d2c0e8a-4b7e-4c01-8d33-100000000003'
    ) OR EXISTS (
        SELECT 1 FROM workspace_members WHERE workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
    ) OR EXISTS (
        SELECT 1 FROM workspace_members WHERE user_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
    ) OR EXISTS (
        SELECT 1 FROM agents WHERE workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
    ) OR EXISTS (
        SELECT 1 FROM webhook_configs WHERE workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
    ) OR EXISTS (
        SELECT 1 FROM usage_limits WHERE workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
    ) OR EXISTS (
        SELECT 1 FROM workspaces
        WHERE owner_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND id <> '7d2c0e8a-4b7e-4c01-8d33-100000000002'
    ) OR EXISTS (
        SELECT 1 FROM tasks
        WHERE created_by_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND id <> '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM activity_log
        WHERE (task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
            OR workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
            OR actor_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001')
          AND id <> '7d2c0e8a-4b7e-4c01-8d33-100000000006'
    ) THEN
        RAISE EXCEPTION 'pilot rollback blocked by non-pilot dependents';
    END IF;
END;
$rollback_preflight$;

DELETE FROM activity_log
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000006'
  AND task_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000004'::uuid
  AND workspace_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000002'::uuid
  AND actor_type IS NOT DISTINCT FROM 'agent'
  AND actor_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000001'::uuid
  AND action IS NOT DISTINCT FROM 'task.created'
  AND payload IS NOT DISTINCT FROM '{"source":"LTM-0025","mode":"pilot"}'::jsonb
  AND created_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz;

DELETE FROM task_checklists
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000005'
  AND task_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000004'::uuid
  AND text IS NOT DISTINCT FROM 'Pilot task graph is proven in the KB through the granted principal'
  AND checked IS NOT DISTINCT FROM false
  AND position IS NOT DISTINCT FROM 1
  AND created_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz;

DELETE FROM task_tags
WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
  AND tag IS NOT DISTINCT FROM 'graph-merge';

DELETE FROM task_tags
WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
  AND tag IS NOT DISTINCT FROM 'pilot';

DELETE FROM tasks
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
  AND project_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000003'::uuid
  AND sprint_id IS NULL AND parent_id IS NULL
  AND title IS NOT DISTINCT FROM 'LTM-0025 Muneral to KB graph-merge pilot'
  AND description IS NOT DISTINCT FROM 'Prove deterministic task graph ingestion, recall, and idempotent replay.'
  AND status IS NOT DISTINCT FROM 'in_progress'
  AND priority IS NOT DISTINCT FROM 'high'
  AND due_date IS NULL
  AND estimate_hours IS NOT DISTINCT FROM 2.00
  AND created_by_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000001'::uuid
  AND actor_type IS NOT DISTINCT FROM 'agent'
  AND created_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz
  AND updated_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz;

DELETE FROM projects
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
  AND workspace_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000002'::uuid
  AND slug IS NOT DISTINCT FROM 'long-term-memory'
  AND name IS NOT DISTINCT FROM 'Long Term Memory'
  AND description IS NOT DISTINCT FROM 'Deterministic Muneral to KB graph-merge pilot project.'
  AND repo_url IS NULL
  AND created_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz;

DELETE FROM workspaces
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
  AND slug IS NOT DISTINCT FROM 'ltm-0025-pilot'
  AND name IS NOT DISTINCT FROM 'LTM-0025 Pilot'
  AND owner_id IS NOT DISTINCT FROM '7d2c0e8a-4b7e-4c01-8d33-100000000001'::uuid
  AND subscription_tier IS NOT DISTINCT FROM 'free'
  AND created_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz;

DELETE FROM users
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
  AND github_id IS NULL AND telegram_id IS NULL
  AND name IS NOT DISTINCT FROM 'LTM-0025 Pilot Agent'
  AND avatar_url IS NULL
  AND created_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz
  AND updated_at IS NOT DISTINCT FROM '2026-07-15T00:00:00Z'::timestamptz;

COMMIT;
