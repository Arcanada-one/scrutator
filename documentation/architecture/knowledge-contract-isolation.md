# Knowledge Contract Arcana storage isolation

## Decision and evidence boundary

Use a separate Scrutator application instance and a separate PostgreSQL database
for the Knowledge Contract Arcana corpus. Give that instance its own database
role, service credentials, runtime configuration, deployment transaction, health
check, and backup/restore exercise. The existing shared instance and its tenants
remain on their current deployment path.

The current Scrutator code has one module-global connection pool created from
one database URL. A namespace cannot select a second database through that pool.
The shared instance also uses one application database role; namespace filtering
in application queries is not a database routing boundary. Changing its URL
would move every existing tenant. A project namespace in that instance is
therefore insufficient for the required project storage and role isolation. This
decision should be revisited only after a reviewed routing seam proves tenant-
safe behavior and equivalent operational isolation.

This document specifies a migration plan. It does not attest that the project
instance, database, role, route, or corpus has been deployed.

## Source and deployment sequence

1. Record the running shared instance's source revision, database target, tenant
   count, health, and backup baseline without printing credentials. Confirm that
   the project database and role names are unused.
2. Add a project-only deployment definition and transaction in source control.
   Its bind must initially be loopback or a dedicated mesh address with an
   explicit ingress rule; it must not claim or change the shared instance's
   listener. Keep separate Compose project names, volumes, environment files,
   image tags, and rollback state. Validate the actual host network exposure
   before enabling a route.
3. Create a dedicated PostgreSQL database and non-superuser login role through a
   reviewed, idempotent provisioning procedure. Install required extensions and
   schema only in that database. Grant no table, sequence, or schema rights in
   the shared database. Keep the password in protected storage, never in source
   or CI logs.
4. Start the project instance against the dedicated database. Create only its
   project namespace and scoped feeder/query principals. The shared instance's
   configuration, database URL, namespace list, and credentials must stay byte-
   identical across the change.
5. Prove isolation with positive project reads/writes and negative attempts by
   the project role and service principal against a named existing tenant. Test
   unauthenticated and wrong-namespace API requests. A failed probe without a
   positive control is not proof of isolation.
6. Admit a source generation only after allowlist, license and secret checks,
   source-to-index digest/count parity, and a draft negative control. A healthy
   service with an empty database is not an ingest success.
7. Measure a backup snapshot containing the new database, then restore into an
   isolated target and record recovery point and duration. The existing database
   backup path is an input to this test, not restore evidence by itself.

## Reversal and stop conditions

If provisioning, isolation, capacity, backup, or restore checks fail, leave the
shared instance untouched, stop the project instance, and keep its route
disabled. Preserve the new database and logs for diagnosis; database deletion
needs a separate decision. If a new index generation fails, keep the last known
good project generation and record the gap. A project rollback must not restart
or reconfigure the shared Scrutator deployment.

Do not promote the project route while the corpus is empty, publication rights
are unknown, cross-tenant negative controls fail, the project role can read
shared tables, or a backup restore has not been measured. This is a reversible
deployment design, not a production readiness verdict.
