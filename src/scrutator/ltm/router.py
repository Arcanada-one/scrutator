"""FastAPI router for LTM endpoints."""

from __future__ import annotations

import logging
import secrets
import time

from fastapi import APIRouter, Depends, Header, HTTPException

from scrutator.auth.dependency import require_tenant_context, resolve_namespace_selector
from scrutator.auth.models import TenantContext
from scrutator.config import settings
from scrutator.db import repository
from scrutator.ltm.llm import LtmLlmClient
from scrutator.ltm.models import (
    IngestRequest,
    IngestResponse,
    JobStatus,
    LtmJob,
    RecallRequest,
    RecallResponse,
    RecallResult,
    ReflectRequest,
    ReflectResponse,
)
from scrutator.ltm.pipeline import IngestPipeline, RecallPipeline
from scrutator.ltm.reflect import ReflectBudget, ReflectJob
from scrutator.search.searcher import search

log = logging.getLogger("scrutator.ltm.router")

router = APIRouter(prefix="/v1/ltm", tags=["ltm"])


def _create_llm_client() -> LtmLlmClient:
    return LtmLlmClient(
        mc_url=settings.ltm_mc_url,
        connector=settings.ltm_connector,
        model=settings.ltm_model,
        api_key=settings.ltm_mc_api_key,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    ctx: TenantContext = Depends(require_tenant_context),
    x_ltm_writer_token: str | None = Header(default=None),
) -> IngestResponse:
    """Ingest a document: chunk, embed, extract entities/edges.

    Write path — namespace auto-provisioning (`upsert_namespace`) is intentional here (SRCH-0023
    Step 3 keeps it only at write/ingest sites).
    """
    # Reader grants never imply mutation authority. Both generic and
    # structured LTM ingest require this dedicated writer credential.
    try:
        configured_token = settings.ltm_writer_token.encode("ascii")
        supplied_token = x_ltm_writer_token.encode("ascii") if x_ltm_writer_token else b""
    except UnicodeEncodeError:
        valid_writer = False
    else:
        valid_writer = (
            bool(configured_token) and bool(supplied_token) and secrets.compare_digest(configured_token, supplied_token)
        )
    if not valid_writer:
        raise HTTPException(status_code=401, detail="LTM writer credential required")
    allowed_namespaces = {name.strip() for name in settings.ltm_writer_namespaces.split(",") if name.strip()}
    if req.namespace not in allowed_namespaces:
        raise HTTPException(status_code=403, detail="namespace outside LTM writer scope")

    if req.structured_graph is not None:
        namespace_id = await repository.get_namespace_id(req.namespace)
        if namespace_id is not None:
            current_hash = await repository.get_structured_graph_hash(namespace_id, req.source_path)
            if current_hash == req.structured_graph.content_hash:
                return IngestResponse(
                    job_id=f"noop:{current_hash}",
                    status=JobStatus.DONE,
                    idempotent_noop=True,
                )
        else:
            namespace_id = await repository.upsert_namespace(req.namespace)
    else:
        namespace_id = await repository.upsert_namespace(req.namespace)

    if req.project:
        await repository.upsert_project(namespace_id, req.project)

    job_id = await repository.create_ltm_job(namespace_id, req.source_path)

    # Index content using existing Scrutator chunker + embedder
    from scrutator.search.indexer import index_document

    try:
        prior_provenance = None
        if req.structured_graph is not None:
            prior_provenance = await repository.get_source_graph_provenance(namespace_id, req.source_path)
        await repository.update_ltm_job(job_id, status="chunking", current_step="chunking")
        index_result = await index_document(
            content=req.content,
            source_path=req.source_path,
            namespace=req.namespace,
            project=req.project,
        )
        total_chunks = index_result.chunks_indexed

        if req.structured_graph is not None:
            chunk_ids = await repository.get_chunk_ids_by_source(req.source_path, namespace_id)
            graph = req.structured_graph
            result = await repository.apply_structured_graph(
                namespace_id=namespace_id,
                source_path=req.source_path,
                content_hash=graph.content_hash,
                entities=[entity.model_dump() for entity in graph.entities],
                edges=[edge.model_dump() for edge in graph.edges],
                source_chunk_id=chunk_ids[0] if chunk_ids else None,
                prior_entity_ids=prior_provenance["entity_ids"],
                prior_edge_ids=prior_provenance["edge_ids"],
            )
            await repository.update_ltm_job(
                job_id,
                status="done",
                current_step="complete",
                total_chunks=total_chunks,
                processed_chunks=total_chunks,
            )
            return IngestResponse(job_id=job_id, status=JobStatus.DONE, **result)

        await repository.update_ltm_job(
            job_id,
            status="extracting",
            current_step="entity_extraction",
            total_chunks=total_chunks,
        )

        # Get chunk IDs for the indexed document
        chunk_ids = await repository.get_chunk_ids_by_source(req.source_path, namespace_id)

        # Sequential entity/edge extraction per chunk
        llm = _create_llm_client()
        pipeline = IngestPipeline(
            llm=llm,
            namespace=req.namespace,
            namespace_id=namespace_id,
            max_entities_per_chunk=settings.ltm_max_entities_per_chunk,
        )

        from scrutator.db.connection import get_pool

        pool = await get_pool()
        for i, chunk_id in enumerate(chunk_ids):
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT content FROM chunks WHERE id = $1::uuid", chunk_id)
            if row:
                await pipeline.process_chunk(chunk_id, row["content"])
            await repository.update_ltm_job(job_id, processed_chunks=i + 1)

        # Dedup: collect all entity names, ask LLM to group aliases
        await repository.update_ltm_job(job_id, status="deduping", current_step="entity_dedup")
        all_entity_names = await repository.get_entity_names_for_namespace(namespace_id)
        if len(all_entity_names) >= 2:
            dedup_groups = await pipeline.dedup_entities(all_entity_names)
            for group in dedup_groups:
                await repository.merge_entity_aliases(namespace_id, group["canonical"], group.get("aliases", []))

        await repository.update_ltm_job(job_id, status="done", current_step="complete")

    except Exception as exc:
        log.exception("Ingest failed for job %s", job_id)
        await repository.update_ltm_job(job_id, status="failed", error=str(exc)[:500])
        raise HTTPException(status_code=500, detail="Ingest failed") from exc

    return IngestResponse(job_id=job_id, status=JobStatus.DONE)


@router.get("/jobs/{job_id}", response_model=LtmJob)
async def get_job(job_id: str, ctx: TenantContext = Depends(require_tenant_context)) -> LtmJob:
    """Get job status — scoped to the caller's allowed-namespace set."""
    job = await repository.get_ltm_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["namespace_id"] not in ctx.allowed_namespace_ids:
        raise HTTPException(status_code=403, detail="job not in caller's allowed namespace set")
    return LtmJob(
        id=job["id"],
        namespace=str(job["namespace_id"]),
        source_path=job["source_path"],
        status=JobStatus(job["status"]),
        current_step=job.get("current_step"),
        total_chunks=job.get("total_chunks", 0),
        processed_chunks=job.get("processed_chunks", 0),
        error=job.get("error"),
    )


def _search_results_to_dicts(results: list) -> list[dict]:
    return [
        {
            "chunk_id": r.chunk_id,
            "content": r.content,
            "source_path": r.source_path,
            "score": r.score,
            "namespace": r.namespace,
            "project": r.project,
            "metadata": r.metadata,
        }
        for r in results
    ]


def _dicts_to_recall_results(dicts: list[dict]) -> list[RecallResult]:
    return [
        RecallResult(
            chunk_id=r["chunk_id"],
            content=r["content"],
            source_path=r["source_path"],
            score=r["score"],
            namespace=r["namespace"],
            project=r.get("project"),
            metadata=r.get("metadata", {}),
        )
        for r in dicts
    ]


@router.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest, ctx: TenantContext = Depends(require_tenant_context)) -> RecallResponse:
    """Recall memories with entity enrichment + optional temporal filter."""
    start = time.monotonic()
    namespace_id = await resolve_namespace_selector(ctx, req.namespace)

    # Fetch a wider candidate pool when temporal filter is active to keep `limit` results
    fetch_limit = req.limit * 3 if (req.as_of or req.time_range) else req.limit
    search_response = await search(
        query=req.query, namespace_id=namespace_id, limit=fetch_limit, min_score=req.min_score
    )
    results_dicts = _search_results_to_dicts(search_response.results)

    llm = _create_llm_client()
    pipeline = RecallPipeline(llm=llm, namespace=req.namespace or "arcanada", namespace_id=namespace_id)

    # LTM-0012 — temporal pre-filter (before entity enrichment to save work)
    if req.as_of is not None or req.time_range is not None:
        results_dicts = await pipeline.filter_temporal(
            results=results_dicts, as_of=req.as_of, time_range=req.time_range
        )

    if req.expand_entities and results_dicts:
        enriched = await pipeline.enrich_with_entities(results_dicts)
        # Optional temporal boost in rerank (LTM-0012)
        if req.temporal_boost > 0.0:
            chunk_ids = [r.chunk_id for r in enriched]
            events_by_chunk = await repository.get_chunk_events_summary(chunk_ids)
            enriched = pipeline.apply_temporal_boost(
                enriched, events_by_chunk=events_by_chunk, boost=req.temporal_boost
            )
        # LTM-0013 — append meta-facts as synthetic candidates (gated by flag)
        if settings.ltm_recall_include_meta_facts:
            from scrutator.search.embedder import embed_single

            try:
                q_embedding = await embed_single(req.query)
            except Exception:
                log.exception("embed_single failed for recall meta-fact enrichment")
                q_embedding = None
            enriched = await pipeline.enrich_with_meta_facts(
                enriched,
                query_embedding=q_embedding,
                score_factor=settings.ltm_recall_meta_fact_score_factor,
            )
        enriched = await pipeline.rerank(query=req.query, results=enriched)
    else:
        enriched = _dicts_to_recall_results(results_dicts)

    enriched = enriched[: req.limit]
    elapsed = (time.monotonic() - start) * 1000
    return RecallResponse(results=enriched, total=len(enriched), query=req.query, search_time_ms=round(elapsed, 2))


@router.get("/entities")
async def list_entities(
    namespace: str = "arcanada",
    limit: int = 100,
    ctx: TenantContext = Depends(require_tenant_context),
) -> dict:
    """List entities in a namespace."""
    namespace_id = await resolve_namespace_selector(ctx, namespace)
    entities = await repository.list_entities(namespace_id, min(limit, 500))
    return {"entities": entities, "total": len(entities), "namespace": namespace}


@router.get("/graph")
async def get_graph(
    namespace: str = "arcanada",
    entity_name: str | None = None,
    ctx: TenantContext = Depends(require_tenant_context),
) -> dict:
    """Get entity graph (nodes + edges) for a namespace, optionally centered on an entity."""
    namespace_id = await resolve_namespace_selector(ctx, namespace)
    nodes, edges = await repository.get_entity_graph(namespace_id, entity_name)
    return {"nodes": nodes, "edges": edges, "namespace": namespace}


@router.post("/reflect", response_model=ReflectResponse)
async def reflect(req: ReflectRequest, ctx: TenantContext = Depends(require_tenant_context)) -> ReflectResponse:
    """LTM-0013 — manual trigger for one reflect run."""
    if not settings.ltm_reflect_enabled:
        raise HTTPException(status_code=503, detail="reflect disabled by config")
    namespace_id = await resolve_namespace_selector(ctx, req.namespace)
    llm = _create_llm_client()
    budget = ReflectBudget(
        max_usd=settings.ltm_reflect_budget_usd,
        max_req=settings.ltm_reflect_budget_req_count,
    )
    job = ReflectJob(
        llm=llm,
        namespace=req.namespace,
        namespace_id=namespace_id,
        budget=budget,
        max_meta_facts_per_group=settings.ltm_reflect_max_meta_facts_per_chunk,
    )
    summary, facts = await job.run(
        since=req.since,
        max_chunks=req.max_chunks,
        dry_run=req.dry_run,
    )
    return ReflectResponse(
        summary=summary,
        preview=facts if req.dry_run else None,
    )


@router.get("/meta_facts")
async def list_meta_facts(
    namespace: str = "arcanada",
    fact_type: str | None = None,
    limit: int = 100,
    ctx: TenantContext = Depends(require_tenant_context),
) -> dict:
    """LTM-0013 — debug listing of meta-facts in a namespace."""
    namespace_id = await resolve_namespace_selector(ctx, namespace)
    facts = await repository.list_meta_facts_by_namespace(
        namespace_id=namespace_id,
        fact_type=fact_type,
        limit=min(max(limit, 1), 500),
    )
    return {"meta_facts": facts, "total": len(facts), "namespace": namespace}


@router.get("/events")
async def list_events(
    entity: str,
    namespace: str = "arcanada",
    include_superseded: bool = False,
    ctx: TenantContext = Depends(require_tenant_context),
) -> dict:
    """LTM-0012 — list temporal events for an entity in a namespace."""
    namespace_id = await resolve_namespace_selector(ctx, namespace)
    events = await repository.get_events_for_entity(
        namespace_id=namespace_id,
        entity_name=entity,
        include_superseded=include_superseded,
    )
    return {"events": events, "total": len(events), "entity": entity, "namespace": namespace}
