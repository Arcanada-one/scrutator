#!/usr/bin/env python3
"""KB-CONS-02 one-host, idempotent arcana-dbs root provisioning transaction for the kc-corpus instance.

Mirrors the reviewed KCP-0017 provisioner (contract-place-api deploy/knowledge-contract-scrutator).

Control runs this from a reviewed resulting-main checkout. It never accepts a
database name, role, SQL, host, Vault path or password from its caller.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DOCKER = "/usr/bin/docker"
CONTAINER = "arcana-postgres"
HBA_CONTAINER_PATH = "/var/lib/postgresql/18/docker/pg_hba.conf"
HBA_VOLUME_ROOT = Path("/var/lib/docker/volumes/arcana-db_pg-data/_data")
HBA_HOST_PATH = HBA_VOLUME_ROOT / "18/docker/pg_hba.conf"
HBA_BACKUP = Path("/root/kbcons02-pg-hba-before-kc-corpus.conf")
VAULT_ADDR = "http://100.97.136.74:8200"
VAULT_PATH = "/v1/arcanada/data/projects/kc-corpus/scrutator-db"
ROLE = "kc_corpus_scrutator_app"
DATABASE = "kc_corpus"
SHARED_DATABASE = "arcanada_scrutator"
KC_DATABASE = "knowledge_contract"
KC_BEGIN = "# BEGIN KCP-0017 dedicated Scrutator DB deny\n"
KC_END = "# END KCP-0017 dedicated Scrutator DB deny\n"
HBA_BLOCK = (
    "# BEGIN KB-CONS-02 kc-corpus Scrutator DB deny\n"
    "local arcanada_scrutator kc_corpus_scrutator_app reject\n"
    "host arcanada_scrutator kc_corpus_scrutator_app all reject\n"
    "local knowledge_contract kc_corpus_scrutator_app reject\n"
    "host knowledge_contract kc_corpus_scrutator_app all reject\n"
    "host kc_corpus kc_corpus_scrutator_app 127.0.0.1/32 scram-sha-256\n"
    "# END KB-CONS-02 kc-corpus Scrutator DB deny\n"
)
DOCKER_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/root", "LANG": "C",
              "DOCKER_HOST": "unix:///var/run/docker.sock"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def docker_call(arguments: list[str], sql: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([DOCKER, *arguments], input=sql, text=True, capture_output=True,
                          check=False, timeout=30, env=DOCKER_ENV)


def psql(database: str, sql: str, *, role: str = "postgres", expected_failure: bool = False) -> str:
    command = ["exec", "-i", CONTAINER, "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1",
               "-U", role, "-d", database, "-At"]
    result = docker_call(command, sql)
    if expected_failure:
        if result.returncode == 0 or "pg_hba.conf rejects connection" not in result.stderr:
            fail("shared database negative probe did not prove an HBA rejection")
        return ""
    if result.returncode != 0:
        # psql can echo the failed SQL (including a password). Keep all captured
        # output private and report only the phase, never stderr or SQL.
        fail("PostgreSQL operation failed; inspect protected DB logs")
    return result.stdout.strip()


def vault_request(method: str, token: str, payload: dict | None = None) -> dict | None:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(VAULT_ADDR + VAULT_PATH, data=body, method=method,
                                     headers={"X-Vault-Token": token, "Content-Type": "application/json"})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=8) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if method == "GET" and error.code == 404:
            return None
        fail(f"Vault {method} failed with HTTP {error.code}")
    except (OSError, ValueError):
        fail("Vault request failed or response was malformed")


def obtain_password(token: str) -> str:
    existing = vault_request("GET", token)
    if existing is not None:
        data = existing.get("data", {}).get("data", {})
        if data.get("username") != ROLE or data.get("database") != DATABASE:
            fail("Vault project credential path has a conflicting identity")
        password = data.get("password")
        if not isinstance(password, str) or not re.fullmatch(r"[A-Za-z0-9_-]{40,}", password):
            fail("Vault project credential is malformed")
        return password
    password = secrets.token_urlsafe(48)
    vault_request("POST", token, {"options": {"cas": 0},
                                  "data": {"username": ROLE, "database": DATABASE,
                                           "password": password}})
    # Readback verifies the KV v2 write without displaying the value.
    written = vault_request("GET", token)
    if written is None or written.get("data", {}).get("data", {}).get("password") != password:
        fail("Vault credential readback failed")
    return password


def install_hba_deny() -> None:
    mount = docker_call(["inspect", CONTAINER, "--format",
                         '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Source}}{{end}}{{end}}'])
    if mount.returncode != 0 or mount.stdout.strip() != str(HBA_VOLUME_ROOT):
        fail("PostgreSQL data mount differs from reviewed host mapping")
    if psql("postgres", "SHOW hba_file;\n") != HBA_CONTAINER_PATH:
        fail("PostgreSQL HBA path differs from reviewed host mapping")
    for path in (HBA_HOST_PATH, *HBA_HOST_PATH.parents):
        if path.is_symlink():
            fail("HBA path contains a symlink")
    metadata = HBA_HOST_PATH.stat()
    if not stat.S_ISREG(metadata.st_mode):
        fail("HBA source is not a regular file")
    original = HBA_HOST_PATH.read_text()
    # The reject rules must precede the generic `host all all all` rule. The reviewed KCP-0017 block is
    # first in the file; this block goes directly after it and never moves it.
    if original.count(KC_BEGIN) != 1 or original.count(KC_END) != 1:
        fail("KCP-0017 HBA block is missing or duplicated; refusing to place the kc-corpus block")
    anchor = original.index(KC_END) + len(KC_END)
    if HBA_BLOCK in original:
        if original.count(HBA_BLOCK) != 1 or original[anchor:anchor + len(HBA_BLOCK)] != HBA_BLOCK:
            fail("installed kc-corpus HBA block is not directly after the KCP-0017 block")
        if not HBA_BACKUP.is_file() or HBA_BACKUP.is_symlink() or \
                HBA_BACKUP.read_text() != original[:anchor] + original[anchor + len(HBA_BLOCK):]:
            fail("installed HBA block has no matching original backup")
    else:
        if "BEGIN KB-CONS-02" in original or "END KB-CONS-02" in original:
            fail("partial or changed KB-CONS-02 HBA block")
        if HBA_BACKUP.exists():
            fail("HBA backup exists without installed block; inspect before retry")
        backup_fd = os.open(HBA_BACKUP, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(backup_fd, "w") as backup:
            backup.write(original)
            backup.flush()
            os.fsync(backup.fileno())
        temporary_fd, temporary_name = tempfile.mkstemp(prefix=".kbcons02-hba-", dir=HBA_HOST_PATH.parent)
        try:
            os.fchmod(temporary_fd, stat.S_IMODE(metadata.st_mode))
            os.fchown(temporary_fd, metadata.st_uid, metadata.st_gid)
            with os.fdopen(temporary_fd, "w") as temporary:
                temporary.write(original[:anchor] + HBA_BLOCK + original[anchor:])
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, HBA_HOST_PATH)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    if psql("postgres", "SELECT pg_reload_conf();\n") != "t":
        fail("PostgreSQL HBA reload was not accepted")
    if psql("postgres", "SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL;\n") != "0":
        fail("PostgreSQL HBA has a parse error after reload")
    rules = psql("postgres", "SELECT type, database[1], auth_method FROM pg_hba_file_rules "
                 f"WHERE user_name @> ARRAY['{ROLE}']::text[] "
                 "AND auth_method='reject' ORDER BY line_number;\n")
    if rules.splitlines() != ["local|arcanada_scrutator|reject", "host|arcanada_scrutator|reject",
                              "local|knowledge_contract|reject", "host|knowledge_contract|reject"]:
        fail("dedicated role HBA deny rules were not read back")
    allowed = psql("postgres", "SELECT type, auth_method FROM pg_hba_file_rules "
                   f"WHERE database @> ARRAY['{DATABASE}']::text[] "
                   f"AND user_name @> ARRAY['{ROLE}']::text[] "
                   "AND auth_method='scram-sha-256' ORDER BY line_number;\n")
    if allowed != "host|scram-sha-256":
        fail("dedicated role TCP password rule was not read back")
    kc_rules = psql("postgres", "SELECT count(*) FROM pg_hba_file_rules "
                    "WHERE user_name @> ARRAY['kc_scrutator_app']::text[] AND error IS NULL;\n")
    if kc_rules != "3":
        fail("KCP-0017 role HBA rules changed")


def tcp_probe(database: str, password: str, *, expected_failure: bool = False) -> str:
    # A .pgpass record arrives through Docker stdin, never argv, and is deleted
    # by the child trap. The query itself is fixed and carries no secret.
    script = ('set -eu; umask 077; passfile=$(mktemp /tmp/kbcons02-pgpass.XXXXXX); '
              'trap \'rm -f "$passfile"\' EXIT HUP INT TERM; '
              'cat > "$passfile"; '
              'PGPASSFILE="$passfile" psql -X -q -v ON_ERROR_STOP=1 '
              '-h 127.0.0.1 -U ' + ROLE + ' -d ' + database +
              ' -At -c "SELECT current_user || chr(124) || current_database()"')
    result = docker_call(["exec", "-i", CONTAINER, "sh", "-c", script],
                         f"127.0.0.1:5432:*:{ROLE}:{password}\n")
    if expected_failure:
        if result.returncode == 0 or "pg_hba.conf rejects connection" not in result.stderr:
            fail("shared database TCP negative probe did not prove HBA rejection")
        return ""
    if result.returncode != 0:
        fail("dedicated database password-authenticated TCP probe failed")
    return result.stdout.strip()


def provision(password: str) -> None:
    # The generated password is sent through stdin to psql and never argv.
    # Re-running reuses the existing Vault value and converges the role.
    existing_owner = psql("postgres", f"SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname='{DATABASE}';\n")
    if existing_owner and existing_owner != ROLE:
        fail("project database exists with another owner")
    membership_count = psql("postgres", f"SELECT count(*) FROM pg_auth_members "
                            f"WHERE member=(SELECT oid FROM pg_roles WHERE rolname='{ROLE}');\n")
    if membership_count != "0":
        fail("project role has pre-existing memberships")
    role_sql = f"""
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ROLE}') THEN
    CREATE ROLE {ROLE} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
  END IF;
END $$;
ALTER ROLE {ROLE} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS PASSWORD {sql_literal(password)};
SELECT format('CREATE DATABASE %I OWNER %I', '{DATABASE}', '{ROLE}')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '{DATABASE}') \\gexec
"""
    psql("postgres", role_sql)
    owner = psql("postgres", f"SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname='{DATABASE}';\n")
    if owner != ROLE:
        fail("project database exists with another owner")
    psql("postgres", f"REVOKE ALL ON DATABASE {DATABASE} FROM PUBLIC;\n"
                     f"GRANT CONNECT, TEMPORARY ON DATABASE {DATABASE} TO {ROLE};\n")
    psql(DATABASE, "CREATE EXTENSION IF NOT EXISTS vector;\n"
                   "CREATE EXTENSION IF NOT EXISTS btree_gist;\n")
    extension_count = psql(DATABASE, "SELECT count(*) FROM pg_extension "
                            "WHERE extname IN ('vector','btree_gist');\n")
    if extension_count != "2":
        fail("project extensions were not read back")
    positive = tcp_probe(DATABASE, password)
    if positive != f"{ROLE}|{DATABASE}":
        fail("project role/database positive probe failed")
    tcp_probe(SHARED_DATABASE, password, expected_failure=True)
    tcp_probe(KC_DATABASE, password, expected_failure=True)


def main() -> int:
    if len(sys.argv) != 1 or os.geteuid() != 0:
        fail("run with no arguments as control root on arcana-dbs")
    if os.environ.get("VAULT_ADDR") != VAULT_ADDR:
        fail("VAULT_ADDR must be the reviewed arcana-dbs primary")
    token = os.environ.get("VAULT_TOKEN", "")
    if not token or "\n" in token or "\r" in token:
        fail("a Vault operator token is required in the protected environment")
    if psql("postgres", "SELECT current_user;\n") != "postgres":
        fail("unexpected PostgreSQL control identity")
    password = obtain_password(token)
    install_hba_deny()
    provision(password)
    print(f"DB_PROVISION_PASS database={DATABASE} role={ROLE} extensions=vector,btree_gist")
    print(f"SHARED_DB_DENY_PASS role={ROLE} databases={SHARED_DATABASE},{KC_DATABASE} method=pg_hba_reject")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        # Never include exception details: URL/SQL/captured child output may
        # contain the credential. A failed phase is investigated in protected
        # control logs, not in a CI log or public PR.
        print("DB_PROVISION_FAIL: " + (str(error) if isinstance(error, RuntimeError) else "I/O failure"),
              file=sys.stderr)
        raise SystemExit(1)
