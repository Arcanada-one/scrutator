"""Configuration for ML vs LLM benchmark (SRCH-0014)."""

import os

# --- Scrutator API (via Tailscale) ---
SCRUTATOR_URL = os.environ.get("SCRUTATOR_URL", "http://100.70.137.104:8310")

# --- OpenRouter (for LLM baselines: Haiku, GPT-4o-mini) ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Models ---
HAIKU_MODEL = "anthropic/claude-3.5-haiku"
GPT4OMINI_MODEL = "openai/gpt-4o-mini"
PHI4_OLLAMA_MODEL = "phi4-mini"
GLINER2_MODEL = "fastino/gliner2-multi-v1"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# --- Benchmark parameters ---
SAMPLE_SIZE = 100
MANUAL_ANNOTATION_SIZE = 30
ENTITY_TYPES = [
    "person",
    "project",
    "concept",
    "technology",
    "event",
    "organization",
    "location",
]

# --- Area classification labels (from Agent Dreamer) ---
AREA_LABELS = ["AI", "arcana", "security", "business", "tools", "philosophy"]

# --- LLM pricing (USD per 1M tokens, as of 2026-04) ---
PRICING = {
    HAIKU_MODEL: {"input": 0.80, "output": 4.00},
    GPT4OMINI_MODEL: {"input": 0.15, "output": 0.60},
    # Phi-4-mini and ML models: $0 (local)
    PHI4_OLLAMA_MODEL: {"input": 0.0, "output": 0.0},
    GLINER2_MODEL: {"input": 0.0, "output": 0.0},
    RERANKER_MODEL: {"input": 0.0, "output": 0.0},
}

# --- Paths ---
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
