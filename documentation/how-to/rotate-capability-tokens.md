# Rotate the feeder write token and the LTM writer token

Task-oriented procedure. Operator-specific paths, hosts and consumer lists belong in the operator's private runbook, not here.
No token value belongs in a terminal, a log, a commit or a ticket.

## What you rotate

| Token | Request header | Setting |
|---|---|---|
| Feeder write | `X-KB-Feeder-Token` | `SCRUTATOR_FEEDER_TOKEN` |
| LTM writer | `X-LTM-Writer-Token` | `SCRUTATOR_LTM_WRITER_TOKEN` |

Each setting holds ONE value, compared in constant time (`secrets.compare_digest`). There is no overlap window: from the moment
the service starts with the new value, every client still presenting the old one gets `401`. Plan for that.

## Constraints of the deploy route

- The service is created from `docker-compose.yml` plus the deploy transaction's candidate override. Do not recreate it by
  hand with `docker-compose.yml` alone; that drops the override. Use the Deploy workflow.
- `deploy/scrutator-deploy-transaction.sh` refuses a Deploy at the commit that is already live (`immutable candidate tag already
  exists`) and aborts when `.env` changes while it runs. So write the new `.env` before the Deploy starts, and give the Deploy a
  new commit. A commit that touches nothing in the deploy surface (for example this page) needs no reviewed-SHA variable.

## Procedure

1. Stop every client that holds the token (timers, workers) and wait for running instances to finish.
2. Back up the settings file and each client's credential file to a root-only directory outside git.
3. Generate each new value on the host (`python3 -c 'import secrets; ...'` written straight to a file with `umask 077`), and
   replace the value in the settings file and in every client credential file, keeping owner and mode. Never print a value.
4. Land a new commit that is outside the deploy surface; the push-triggered Deploy rebuilds and recreates the service with the
   new settings file and runs the canary.
5. Verify with header files passed by reference (`curl -H @file`), never with a value in the command line:
   - new feeder value: `GET /v1/index/capability` answers `200`; old value and no value answer `401`;
   - new LTM value: `POST /v1/ltm/ingest` with no body answers `422` (authenticated, nothing written); old value answers `401`;
   - each token is refused on the other token's route.
6. Start the clients again, then watch their next scheduled runs.
7. Shred the backup directory and search the host for older copies of the old values (deploy snapshots and checkpoints).

## Roll back

Restore the settings file and the client credential files from the backup, then deploy again with another new commit. A rollback
restores service only; the old values stay exposed, so repeat the rotation once the cause is fixed.
