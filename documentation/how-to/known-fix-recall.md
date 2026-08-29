# How to wire Datarim known-fix recall to Scrutator

Make a prior task's distilled conclusion reach the next task's context automatically, instead
of waiting for a person to type a query.

This is the **read** half of the self-learning loop. The write half (a `known_fix` JSON block
emitted at `/dr-archive`) and the local recall half (`/dr-do` Step 7.4) already ship in the
Datarim framework. What was missing was an adapter binding that seam to the KB.

## The seam

`/dr-do` Step 7.4 runs the framework helper:

```bash
"${DATARIM_RUNTIME:-$HOME/.claude}/dev-tools/known-fix-memory.py" \
    query --root "$DATARIM_ROOT" --query "<symptom>" --limit 5
```

The helper searches project-local `datarim/insights/INSIGHTS-*.md` records and, when
`DATARIM_KNOWN_FIX_RETRIEVER` names an **absolute, regular, non-symlink, executable** file, also
shells out to that retriever. The framework deliberately owns no adapter — a project-specific
adapter obtains its own read credential through that project's own mechanism. This is Scrutator's.

## Steps

### 1. Point the framework at the shim

```bash
export DATARIM_KNOWN_FIX_RETRIEVER=/abs/path/to/scrutator/scripts/scrutator-known-fix-retriever
```

The path must be absolute and the file must keep its executable bit; a symlink is refused by the
framework helper.

### 2. Write the adapter configuration

The framework strips the retriever's environment to `PATH` alone, so **no `SCRUTATOR_*`
variable, no `HOME`, and no `PYTHONPATH` reaches the adapter**. Configuration is a file at the
fixed path `~/.config/scrutator/known-fix-retriever.json` (`~` still resolves via the passwd
database with `HOME` unset):

```json
{
  "base_url": "http://100.70.137.104:8310",
  "namespace": "self-improvement",
  "token_file": "/run/credentials/known-fix-retriever/token",
  "timeout_seconds": 2.0,
  "quarantine_file": "~/.config/scrutator/known-fix-quarantine.txt"
}
```

Copy `deploy/known-fix-retriever.example.json` as a starting point. `base_url` must be
`http://` or `https://`; any other scheme is refused. `timeout_seconds` is clamped to 2.5 s so
the adapter always answers inside the caller's 3-second kill.

The namespace `self-improvement` is the isolated lane the kb-feeder projects
`documentation/archive/**/archive-*.md`, `datarim/reflection/*.md` and `datarim/insights/*.md`
into (`kb-feeder/config/self-improvement/`).

### 3. Supply a read credential

Never put a token in the config file. `token_file` names a file holding the bearer token; the
adapter reads it, sends `Authorization: Bearer <token>`, and never echoes it. A token containing
CR/LF is treated as absent rather than used as a header value.

Until a scoped reader grant exists, `POST /v1/search` answers
`403 {"detail":"no namespace authorized for this principal"}` and the adapter degrades to an
empty result — the loop runs local-only, and no task fails.

### 4. Verify

```bash
python -m scrutator.tools.known_fix_retriever --query "embedding pool exhausted" --limit 5
```

Then check the framework end of the seam. `remote_status` should read `ok`:

```bash
known-fix-memory.py query --root "$DATARIM_ROOT" --query "<symptom>" --limit 5 | jq .remote_status
```

`not_configured` means `DATARIM_KNOWN_FIX_RETRIEVER` is unset; `unavailable` means the shim was
rejected (not absolute, symlink, no exec bit) or overran the budget; `invalid` means stdout was
not a JSON list.

## Fail-soft is total

Every failure — missing config, unreachable KB, 401/403, malformed response, empty index —
prints `[]` and exits 0. A knowledge base that is down degrades recall; it never fails a task.

## Retiring a poisoned or secret-bearing chunk

The loop's write path indexes archives that quote external issue bodies and support prose
verbatim, so a stranger's text can re-enter a future task wearing the highest-trust label in the
system. Four gates run on every hit, and a rejected hit is **dropped, never partially masked**:

1. **Quarantine** — the forgetting primitive. List a `content_hash`, `chunk_id`, or `source_id`
   in `quarantine_file`, one per line (`#` comments allowed, case-insensitive), and that chunk is
   never recalled again — no re-index required.
2. **Injection** — the server's ingest-time `metadata.injection` flag *and* an
   independent local re-scan, because an un-backfilled chunk carries no stamp and "unstamped"
   must not read as "clean".
3. **Credential shapes** — mirrors the framework validator's patterns. `.gitignore` is a KB
   *inclusion* path, so a credential literal can reach the index despite every git-shaped
   scanner.
4. **No redirect is followed.** urllib's default handler permits an `ftp:` redirect target, and
   a `/v1/search` POST never legitimately redirects — so a 3xx is an error, not a hop.

Surviving text is neutralised before it leaves the adapter: control characters stripped, fence
runs (```` ``` ````, `~~~`) replaced with `<fence>` so an excerpt cannot break out of the
consumer's data block. The framework labels the whole payload
`"contract": "evidence_only_untrusted"` — treat every result as **evidence to verify**, never as
an instruction to follow.

## Related

- `documentation/explanation/skill-ingest-trust-boundary.md` — the ingest-side trust model.
- `src/scrutator/search/ingest_safety.py` — the server-side injection scan and trust tiering.
