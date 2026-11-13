# Install and roll back a muneral-kb-sync release

`deploy/install-muneral-kb-sync.sh` builds a sealed, immutable release of the Muneral to Scrutator sync under
`/opt/muneral-kb-sync/releases/<sha>` and switches the `current` symlink. This is not the Scrutator service Deploy: the sync runs
from its own release, so a merged change to `tools/muneral_sync/` reaches the host only through this installer.

## Install

From a clean checkout of the merged commit (the installer refuses a dirty tree or a HEAD that differs from `--sha`):

```
sudo deploy/install-muneral-kb-sync.sh install --sha <40-hex merge commit> --source <that checkout>
sudo systemctl reset-failed muneral-kb-sync.service   # only if the unit sits in the start limit from earlier failures
sudo systemctl start muneral-kb-sync.service           # one real run; then watch the next timer runs
```

The installer keeps the previous release as `previous` and restores the timer state it found.

## What the installer checks first, and refuses otherwise

- **The service account is part of the isolation contract**: a non-root user and group of the same name, one primary group and no
  supplementary groups, shell `/usr/sbin/nologin`, and home exactly `/var/lib/muneral-kb-sync`. An existing account with another
  home is refused (exit 77) before anything changes; fix the account, do not bypass the check.
- **The current and the previous release must still be sealed**: every file owned by root, directories `0555`, files `0444`
  (`bin/` and the venv interpreter `0555`), nothing but regular files and directories, and the contents equal to the release's
  `PAYLOAD_SHA256SUMS`. A release that fails this is refused (exit 65). Nothing is written into a release after it is sealed.
  Diagnostics that import a release's modules run from a COPY outside `releases/`, or with `python -B`; a bytecode cache written
  into a sealed release makes the next install fail.

## Roll back

`deploy/install-muneral-kb-sync.sh rollback` switches `current` back to `previous`. The rollback target is therefore the release
installed before the current one; a release that has lost its seal can no longer serve as one. Releases that neither `current` nor
`previous` point to are unused and may be removed once nothing references them (no symlink, no unit, no running process).

## Schedule and start limit

The timer runs every 5 minutes on the wall clock. Every START counts against the unit's start limit, successful ones too, so the
limit (`StartLimitIntervalSec` / `StartLimitBurst`) must admit all timer starts that fit in its window plus one retry;
`tests/test_systemd_timer.py` pins that relation.
