# Scrutator deploy broker — installation

MEASURED 2026-09-19 (KB-028, KB-027). Two things were left behind when
Scrutator moved to arcana-prd, and both must be repaired for a deploy to run.

## 1. Deploy authority (this broker)

Install as root on arcana-prd:

```
install -o root -g root -m 0755 deploy/scrutator-deploy-broker \
  /usr/local/sbin/scrutator-deploy-broker
install -o root -g root -m 0440 deploy/scrutator-deploy-broker.sudoers \
  /etc/sudoers.d/scrutator-deploy
visudo -c
```

The broker is deliberately NOT run from the workflow checkout: gh-triage in
arcanada-workspace records why — "a deploy script that ships itself would let
any change to it take effect with the same push". The copies here are the
source of truth for review and reinstallation; a change to them reaches
production through an ordinary deploy of the previously reviewed script.

## 2. Deploy-state ownership

`/var/lib/scrutator/deploy-state` is owned by uid 1002 — an account that does
not exist on this host. It is a fragment of the old host, where the deploy
user had that id. The transaction script requires the state directory to be
owned by the caller (`[[ -O "$STATE_DIR" ]]`) with mode 0700, so it refuses
before doing anything:

```
chown -R root:root /var/lib/scrutator/deploy-state
```

Neither step is a convenience. Without the first the deploy cannot touch a
root-owned checkout; without the second it refuses on a directory owned by
nobody.

## Why the error message's suggestion was not taken

git suggests `git config --global --add safe.directory /srv/apps/scrutator`.
That does not grant write access to root-owned files, so it converts one clear
failure into a later unclear one — and insofar as it worked, it would declare
a production checkout writable by the CI account because a message suggested a
command.
