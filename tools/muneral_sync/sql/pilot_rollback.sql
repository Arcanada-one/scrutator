-- Remove only the deterministic LTM-0025 pilot rows, in foreign-key-safe
-- order. Do not delete muneral_kb_task_changes: the tasks DELETE trigger must
-- leave the durable deleted=true tombstone for the incremental source sync.
BEGIN;

DELETE FROM activity_log
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000006';

DELETE FROM task_checklists
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000005';

DELETE FROM task_tags
WHERE task_id = '7d2c0e8a-4b7e-4c01-8d33-100000000004'
  AND tag IN ('graph-merge', 'pilot');

DELETE FROM tasks
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000004';

DELETE FROM projects
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000003';

DELETE FROM workspaces
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000002';

DELETE FROM users
WHERE id = '7d2c0e8a-4b7e-4c01-8d33-100000000001';

COMMIT;
