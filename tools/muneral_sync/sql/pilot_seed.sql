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

COMMIT;
