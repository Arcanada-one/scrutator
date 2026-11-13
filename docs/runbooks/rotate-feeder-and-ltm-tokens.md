# Rotating the feeder write token and the LTM writer token

Written 2026-11-13 (Muneral 316fc638) from the first rotation. Names and paths only: no value belongs in this file, in a
terminal, in a log or in a task comment.

## What is rotated

| Token | Header | Container variable | Stores (same value in all) | Consumers |
|---|---|---|---|---|
| Feeder write | `X-KB-Feeder-Token` | `SCRUTATOR_FEEDER_TOKEN` | `/srv/apps/scrutator/.env`; `/etc/kb-feeder/feeder.token` (0600 root) | `kb-reconcile`, `kb-program-reconcile`, `kb-history-datarim-reconcile`, `kb-self-improvement-reconcile`, `kb-kc2-store-reconcile` (all `LoadCredential=feeder-token:`), `kb-revert-run.sh` |
| LTM writer | `X-LTM-Writer-Token` | `SCRUTATOR_LTM_WRITER_TOKEN` | `/srv/apps/scrutator/.env`; `/etc/muneral-kb-sync/ltm-writer-token` | `muneral-kb-sync` (`LoadCredential=ltm-writer-token:`) |

The kc-corpus and knowledge-contract Scrutator stacks on the same host hold their own, different values. The container
runs on the host network (port 8310), there is no proxy in front and its access log starts at the last container recreate,
so unknown consumers cannot be traced from history: sample the log, rotate, and restore a consumer that reports 401.

## Constraints that shape the route

- `require_feeder_capability` and `require_ltm_writer_capability` compare ONE configured value
  (`secrets.compare_digest`). There is no overlap: from the container recreate until every consumer holds the new value,
  the consumers fail with 401. Stop the consumer timers first and start them after verification.
- The container is created from `docker-compose.yml` plus a deploy-state `candidate.override.yml`. Never recreate it by hand
  with `docker-compose.yml` alone: that drops the override. Use the repository's Deploy transaction.
- `deploy/scrutator-deploy-transaction.sh` refuses a Deploy at the SHA that is already live (`immutable candidate tag already
  exists`) and aborts if `.env` changes while it runs (`verify_env_exact`). So: write the new `.env` BEFORE the Deploy
  starts, and give the Deploy a NEW commit that is not in `DEPLOY_SURFACE` (a document), so it needs no reviewed-SHA variable.
- Environment listings by NAME only (`gotchas.md`, "Environment listings of a container").

## Order

1. Preconditions: the consumer timers green; the access-log sample taken; GitHub secrets of this repository checked by name.
2. Stop the six consumer timers (five `kb-*-reconcile`, `muneral-kb-sync`); wait for running instances to finish.
3. Back up the three files to a root-only 0700 directory (`/var/lib/scrutator/rotation-<task>/`), outside git.
4. Generate each new value on the host (`secrets.token_hex(32)`, same length as before), write straight into temporary
   files under `umask 077`, move into place keeping owner and mode. Never print, never put in argv.
5. Merge a commit outside `DEPLOY_SURFACE`; the push-triggered Deploy builds, recreates the container with the new `.env`
   and runs the canary. Wait for it to end green.
6. Verify, per token: an authenticated call that cannot write returns not-401 with the new value; the old value, read from the
   backup file by file reference, returns 401 on one feeder route and one LTM route.
7. Start the six timers; the next scheduled runs of `kb-reconcile` and `muneral-kb-sync` and `kb-timer-watch` are green.
8. Shred the backup directory; locate older copies in kb-feeder deploy snapshots by file name and shred or re-check them.

## Rollback

Restore the three files from the backup directory, then deploy again (another new document commit). Rollback restores
service only; the old values stay exposed, so the rotation is repeated after the cause is fixed.
