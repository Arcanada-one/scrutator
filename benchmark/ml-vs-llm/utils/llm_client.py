"""OpenRouter API client for LLM baselines (Haiku, GPT-4o-mini)."""

import json
import time

import requests
from config import OPENROUTER_API_KEY, OPENROUTER_URL


class OpenRouterClient:
    """Thin wrapper around OpenRouter chat completions API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

    def call(
        self,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> dict:
        """Send a chat completion request.

        Returns:
            {"text": str, "input_tokens": int, "output_tokens": int, "latency_ms": float}
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()

        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return {
            "text": choice,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "latency_ms": round(latency_ms, 1),
        }

    def call_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
    ) -> dict:
        """Call and parse JSON from response. Returns {"parsed": ..., ...metadata}."""
        result = self.call(model, prompt, system, max_tokens)
        text = result["text"].strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array/object in the text
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start_idx = text.find(start_char)
                end_idx = text.rfind(end_char)
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        parsed = json.loads(text[start_idx : end_idx + 1])
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                parsed = None

        result["parsed"] = parsed
        return result

    def extract_entities(self, text: str, model: str) -> dict:
        """NER via LLM using Scrutator prompts.

        Returns: {"entities": [...], "input_tokens", "output_tokens", "latency_ms"}
        """
        system = (
            "You are a knowledge graph entity extractor.\n"
            "Extract named entities from the given text.\n"
            "Return ONLY a JSON array, no other text.\n"
            'Each entity: {"name": "...", "type": "..."}\n'
            "Valid types: person, project, concept, technology, event, organization, location.\n"
            "Maximum 10 entities per chunk."
        )
        user = f"Extract entities from this text:\n\n{text[:4000]}"

        result = self.call_json(model, user, system)
        entities = result["parsed"] if isinstance(result["parsed"], list) else []
        return {
            "entities": entities,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"],
        }

    def extract_relations(self, text: str, entities: list[dict], model: str) -> dict:
        """RE via LLM using Scrutator prompts.

        Returns: {"relations": [...], "input_tokens", "output_tokens", "latency_ms"}
        """
        system = (
            "You are a knowledge graph relationship extractor.\n"
            "Given entities and their source text, identify relationships between them.\n"
            "Return ONLY a JSON array, no other text.\n"
            '{"source": "entity_name", "target": "entity_name", "relation": "verb_phrase"}\n'
            "Use short relation names: works_on, depends_on, part_of, created_by, located_in, related_to, etc."
        )
        entity_list = "\n".join(f"- {e.get('name', '')} ({e.get('type', '')})" for e in entities)
        user = f"Entities found:\n{entity_list}\n\nSource text:\n{text[:4000]}"

        result = self.call_json(model, user, system)
        relations = result["parsed"] if isinstance(result["parsed"], list) else []
        return {
            "relations": relations,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"],
        }

    def classify_area(self, text: str, model: str) -> dict:
        """Classify text into one of 6 areas.

        Returns: {"area": str, "input_tokens", "output_tokens", "latency_ms"}
        """
        system = (
            "You are a text classifier. Classify the given text into exactly one area.\n"
            "Valid areas: AI, arcana, security, business, tools, philosophy\n"
            "Return ONLY the area label, nothing else."
        )
        result = self.call(model, f"Classify this text:\n\n{text[:4000]}", system, max_tokens=20)
        area = result["text"].strip().lower()
        return {
            "area": area,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"],
        }

    def rerank(self, query: str, passages: list[dict], model: str) -> dict:
        """Rerank passages by relevance.

        Args:
            passages: [{"id": str, "content": str}]

        Returns: {"ranked_ids": [...], "input_tokens", "output_tokens", "latency_ms"}
        """
        system = (
            "You are a search result reranker.\n"
            "Given a query and candidate search results, reorder by relevance.\n"
            "Return ONLY a JSON array of IDs in order of relevance, most relevant first."
        )
        candidates = "\n\n".join(
            f"ID: {p['id']}\nContent: {p['content'][:200]}" for p in passages
        )
        user = f"Query: {query}\n\nCandidates:\n{candidates}"

        result = self.call_json(model, user, system)
        ranked = result["parsed"] if isinstance(result["parsed"], list) else []
        return {
            "ranked_ids": ranked,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"],
        }
