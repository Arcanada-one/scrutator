"""LTM-0013 — Reflect layer (R in TEMPR): periodic batch meta-fact derivation."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from scrutator.config import settings
from scrutator.db import repository
from scrutator.ltm.models import FactType, MetaFact, ReflectRunSummary
from scrutator.ltm.prompts import format_reflect_summary

if TYPE_CHECKING:
    from scrutator.ltm.llm import LtmLlmClient

log = logging.getLogger("scrutator.ltm.reflect")


class ReflectBudgetExceeded(Exception):
    """Raised when budget cap (USD or req count) is exceeded."""


class ReflectBudget:
    """USD + request-count cap for one reflect run.

    Free-tier callers report 0.0 USD per request; only `req_count` rises.
    """

    def __init__(self, max_usd: float, max_req: int):
        self.max_usd = max_usd
        self.max_req = max_req
        self.spent_usd = 0.0
        self.req_count = 0

    def check(self) -> None:
        if self.spent_usd >= self.max_usd:
            raise ReflectBudgetExceeded(f"USD cap reached: {self.spent_usd:.6f} >= {self.max_usd}")
        if self.req_count >= self.max_req:
            raise ReflectBudgetExceeded(f"req cap reached: {self.req_count} >= {self.max_req}")

    def charge(self, usd: float) -> None:
        self.spent_usd += usd
        self.req_count += 1


async def _embed_for_meta_fact(text: str) -> list[float] | None:
    """Best-effort embedding for meta-fact storage. Returns None on failure."""
    try:
        from scrutator.search.embedder import embed_single

        return await embed_single(text)
    except Exception:
        log.exception("embed_single failed for meta-fact (size=%d)", len(text))
        return None


class ReflectJob:
    """Batch reflection over recent chunks → meta-facts."""

    def __init__(
        self,
        llm: LtmLlmClient,
        namespace: str,
        namespace_id: int,
        budget: ReflectBudget,
        max_meta_facts_per_group: int = 5,
    ):
        self.llm = llm
        self.namespace = namespace
        self.namespace_id = namespace_id
        self.budget = budget
        self.max_meta_facts_per_group = max_meta_facts_per_group

    async def run(
        self,
        since: datetime | None = None,
        max_chunks: int | None = None,
        dry_run: bool = False,
    ) -> tuple[ReflectRunSummary, list[MetaFact]]:
        """Execute one reflect run. Returns (summary, persisted-or-preview)."""
        started = time.monotonic()
        run_id = await repository.create_reflect_run(
            namespace_id=self.namespace_id,
            model_used=self.llm.model,
        )
        chunks_scanned = 0
        meta_facts: list[MetaFact] = []
        abort_reason: str | None = None
        status = "done"
        try:
            limit = max_chunks or settings.ltm_reflect_max_chunks_per_run
            if settings.ltm_reflect_grouping == "cosine":
                chunk_groups = await repository.fetch_chunks_for_reflect_cosine(
                    namespace_id=self.namespace_id,
                    since=since,
                    limit=limit,
                    threshold=settings.ltm_reflect_cosine_threshold,
                )
            else:
                chunk_groups = await repository.fetch_chunks_for_reflect(
                    namespace_id=self.namespace_id,
                    since=since,
                    limit=limit,
                )
            for entity_name, group in chunk_groups.items():
                try:
                    self.budget.check()
                except ReflectBudgetExceeded as exc:
                    abort_reason = str(exc)
                    status = "aborted"
                    break
                chunks_scanned += len(group)
                if len(group) < 2:
                    continue
                facts = await self._reflect_group(entity_name, group, run_id, dry_run)
                meta_facts.extend(facts)
        except Exception as exc:
            log.exception("ReflectJob failed for run %s", run_id)
            status = "failed"
            abort_reason = str(exc)[:500]
        finally:
            await repository.finalize_reflect_run(
                run_id=run_id,
                status=status,
                chunks_scanned=chunks_scanned,
                meta_facts_created=len(meta_facts),
                cost_usd=self.budget.spent_usd,
                req_count=self.budget.req_count,
                abort_reason=abort_reason,
            )
        elapsed_ms = (time.monotonic() - started) * 1000
        summary = ReflectRunSummary(
            run_id=run_id,
            status=status,
            chunks_scanned=chunks_scanned,
            meta_facts_created=len(meta_facts),
            cost_usd=self.budget.spent_usd,
            req_count=self.budget.req_count,
            abort_reason=abort_reason,
            duration_ms=round(elapsed_ms, 2),
        )
        return summary, meta_facts

    async def _reflect_group(
        self,
        entity_name: str,
        chunks: list[dict],
        run_id: str,
        dry_run: bool,
    ) -> list[MetaFact]:
        system, user = format_reflect_summary(
            chunks=chunks,
            entity_names=[entity_name],
            max_facts=self.max_meta_facts_per_group,
        )
        try:
            raw = await self.llm.extract_json(user, system=system)
        except Exception:
            log.exception("LLM call failed for entity %s", entity_name)
            self.budget.charge(0.0)
            return []
        # Free-tier model — no per-call USD cost. Paid models would inject usage here.
        self.budget.charge(0.0)
        if not isinstance(raw, list):
            return []

        out: list[MetaFact] = []
        for item in raw[: self.max_meta_facts_per_group]:
            fact = self._build_meta_fact(item, chunks, run_id)
            if fact is None:
                continue
            if not dry_run:
                vector = await _embed_for_meta_fact(fact.content)
                fact_id = await repository.insert_meta_fact(
                    namespace_id=self.namespace_id,
                    fact=fact,
                    embedding=vector,
                )
                fact.id = fact_id
            out.append(fact)
        return out

    def _build_meta_fact(
        self,
        item: object,
        chunks: list[dict],
        run_id: str,
    ) -> MetaFact | None:
        if not isinstance(item, dict):
            return None
        fact_type_str = str(item.get("fact_type", "")).strip()
        content = str(item.get("content", "")).strip()
        indexes = item.get("source_chunk_indexes", [])
        if not fact_type_str or not content or not isinstance(indexes, list):
            return None
        try:
            fact_type = FactType(fact_type_str)
        except ValueError:
            return None
        source_ids = [chunks[i]["chunk_id"] for i in indexes if isinstance(i, int) and 0 <= i < len(chunks)]
        if not source_ids:
            return None
        try:
            return MetaFact(
                namespace=self.namespace,
                fact_type=fact_type,
                content=content,
                source_chunk_ids=source_ids,
                depth=1,
                model_used=self.llm.model,
                reflect_run_id=run_id,
            )
        except Exception:
            log.warning("MetaFact validation failed: %s", content[:80])
            return None
