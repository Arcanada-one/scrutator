-- LTM-0025 N=1 pilot. The UUIDs and timestamps are fixed so this script is
-- safe to replay and produces the same canonical source graph every time.
BEGIN;

INSERT INTO users (id, name, created_at, updated_at)
VALUES (
    '7d2c0e8a-4b7e-4c01-8d33-100000000001',
    'LTM-0025 Pilot Agent',
    '2026-07-15T00:00:00Z',
    '2026-07-15T00:00:00Z'
)
ON CONFLICT DO NOTHING;

INSERT INTO workspaces (id, slug, name, owner_id, subscription_tier, created_at)
VALUES (
    '7d2c0e8a-4b7e-4c01-8d33-100000000002',
    'ltm-0025-pilot',
    'LTM-0025 Pilot',
    '7d2c0e8a-4b7e-4c01-8d33-100000000001',
    'free',
    '2026-07-15T00:00:00Z'
)
ON CONFLICT DO NOTHING;

INSERT INTO projects (id, workspace_id, slug, name, description, repo_url, created_at)
VALUES (
    '7d2c0e8a-4b7e-4c01-8d33-100000000003',
    '7d2c0e8a-4b7e-4c01-8d33-100000000002',
    'long-term-memory',
    'Long Term Memory',
    'Deterministic Muneral to KB graph-merge pilot project.',
    NULL,
    '2026-07-15T00:00:00Z'
)
ON CONFLICT DO NOTHING;

INSERT INTO tasks (
    id,
    project_id,
    sprint_id,
    parent_id,
    title,
    description,
    status,
    priority,
    due_date,
    estimate_hours,
    created_by_id,
    actor_type,
    created_at,
    updated_at
)
VALUES (
    '7d2c0e8a-4b7e-4c01-8d33-100000000004',
    '7d2c0e8a-4b7e-4c01-8d33-100000000003',
    NULL,
    NULL,
    'LTM-0025 Muneral to KB graph-merge pilot',
    'Prove deterministic task graph ingestion, recall, and idempotent replay.',
    'in_progress',
    'high',
    NULL,
    2.00,
    '7d2c0e8a-4b7e-4c01-8d33-100000000001',
    'agent',
    '2026-07-15T00:00:00Z',
    '2026-07-15T00:00:00Z'
)
ON CONFLICT DO NOTHING;

INSERT INTO task_tags (task_id, tag)
VALUES ('7d2c0e8a-4b7e-4c01-8d33-100000000004', 'graph-merge')
ON CONFLICT DO NOTHING;

INSERT INTO task_tags (task_id, tag)
VALUES ('7d2c0e8a-4b7e-4c01-8d33-100000000004', 'pilot')
ON CONFLICT DO NOTHING;

INSERT INTO task_checklists (id, task_id, text, checked, position, created_at)
VALUES (
    '7d2c0e8a-4b7e-4c01-8d33-100000000005',
    '7d2c0e8a-4b7e-4c01-8d33-100000000004',
    'Pilot task graph is proven in the KB through the granted principal',
    false,
    1,
    '2026-07-15T00:00:00Z'
)
ON CONFLICT DO NOTHING;

INSERT INTO activity_log (id, task_id, workspace_id, actor_type, actor_id, action, payload, created_at)
VALUES (
    '7d2c0e8a-4b7e-4c01-8d33-100000000006',
    '7d2c0e8a-4b7e-4c01-8d33-100000000004',
    '7d2c0e8a-4b7e-4c01-8d33-100000000002',
    'agent',
    '7d2c0e8a-4b7e-4c01-8d33-100000000001',
    'task.created',
    '{"source":"LTM-0025","mode":"pilot"}'::jsonb,
    '2026-07-15T00:00:00Z'
)
ON CONFLICT DO NOTHING;

-- ON CONFLICT is idempotent only when an existing deterministic row has the
-- exact pilot identity. Any collision or drift aborts and rolls back all rows.
DO $pilot_identity$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM users
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND name = 'LTM-0025 Pilot Agent'
          AND github_id IS NULL
          AND telegram_id IS NULL
          AND avatar_url IS NULL
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
          AND updated_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR NOT EXISTS (
        SELECT 1 FROM workspaces
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
          AND slug = 'ltm-0025-pilot'
          AND name = 'LTM-0025 Pilot'
          AND owner_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND subscription_tier = 'free'
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR NOT EXISTS (
        SELECT 1 FROM projects
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
          AND workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
          AND slug = 'long-term-memory'
          AND name = 'Long Term Memory'
          AND description = 'Deterministic Muneral to KB graph-merge pilot project.'
          AND repo_url IS NULL
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR NOT EXISTS (
        SELECT 1 FROM tasks
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND project_id = '7d2c0e8a-4b7e-4c01-8d33-100000000003'
          AND sprint_id IS NULL
          AND parent_id IS NULL
          AND title = 'LTM-0025 Muneral to KB graph-merge pilot'
          AND description = 'Prove deterministic task graph ingestion, recall, and idempotent replay.'
          AND status = 'in_progress'
          AND priority = 'high'
          AND due_date IS NULL
          AND estimate_hours = 2.00
          AND created_by_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND actor_type = 'agent'
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
          AND updated_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR (
        SELECT count(*) FROM task_tags
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) <> 2 OR NOT EXISTS (
        SELECT 1 FROM task_tags
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004' AND tag = 'graph-merge'
    ) OR NOT EXISTS (
        SELECT 1 FROM task_tags
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004' AND tag = 'pilot'
    ) OR NOT EXISTS (
        SELECT 1 FROM task_checklists
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000005'
          AND task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND text = 'Pilot task graph is proven in the KB through the granted principal'
          AND checked = false
          AND position = 1
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR (
        SELECT count(*) FROM task_checklists
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) <> 1 OR NOT EXISTS (
        SELECT 1 FROM activity_log
        WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000006'
          AND task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
          AND workspace_id = '7d2c0e8a-4b7e-4c01-8d33-100000000002'
          AND actor_type = 'agent'
          AND actor_id = '7d2c0e8a-4b7e-4c01-8d33-100000000001'
          AND action = 'task.created'
          AND payload = '{"source":"LTM-0025","mode":"pilot"}'::jsonb
          AND created_at = '2026-07-15T00:00:00Z'::timestamptz
    ) OR (
        SELECT count(*) FROM activity_log
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) <> 1 OR EXISTS (
        SELECT 1 FROM task_dependencies
        WHERE from_task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
           OR to_task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM task_agents
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM task_git_refs
        WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) OR EXISTS (
        SELECT 1 FROM tasks
        WHERE parent_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
    ) THEN
        RAISE EXCEPTION 'pilot seed identity mismatch';
    END IF;
END;
$pilot_identity$;

COMMIT;
