# FI10 namespaces (`fi10-public` / `fi10-private`)

How to exercise the Scrutator-side FI10 namespace ACL, generation readback and
withdrawal purge **without** enabling live Feeder grants.

## What this covers

- Namespace names: `fi10-public`, `fi10-private`
- Principal × namespace ACL (`decide_read`)
- Generation readback (database/collection, counts, exact generation id, positive control)
- Withdrawal purge + last-known-good / forbidden-fallback rules
- Synthetic harness receipt for FI10-T5.2

## What this does not cover

- Live `SCRUTATOR_FEEDER_NAMESPACES` append (held; needs paired `arcanada-workspace` Feeder change)
- Production namespace creation (`bootstrap_fi10_namespaces.py --live` refuses)
- Production purge
- Binding an accepted `fi10-public-<date>-<n>` tag from FI10-T2.8 (producer input)

## Synthetic harness

```bash
PYTHONPATH=src python3 scripts/fi10_namespace_readback.py \
  --receipt /tmp/fi10-namespace-harness-receipt.json
```

Exit `0` only when all synthetic AC controls pass. `--live` is refused.

Preview-only bootstrap (name validation; no DB write):

```bash
python3 scripts/bootstrap_fi10_namespaces.py
```

## Activation (later; not this change)

1. FI10-T2.8 publishes a consumable generation tag + manifest sha256.
2. FI10-T5.4 publishes a Feeder dry-run receipt against that generation.
3. Paired Feeder grant + Scrutator compose append (same pattern as other projected/runtime scopes).
4. Operator `grant_namespace.py` for reader principals; never create namespaces as a side effect of a grant.
