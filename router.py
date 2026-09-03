"""
Hybrid model router.

Decides whether a question should be handled by the local Ollama model
(fast, free, good enough for simple lookups) or by a cloud model via
OpenRouter (slower, paid, better at complex reasoning).

Two-tier design:
  Tier 1 — instant heuristics (rules). Catches the obvious cases at
           zero cost: simple lookups -> local, clear synthesis tasks -> cloud.
  Tier 2 — ambiguous questions get one cheap classification call to the
           local model itself. If the local model is down or gives a
           bad answer, we fail safe to the cloud.
"""
import os
import re
import time
import requests

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
LOCAL_MODEL = os.environ.get('LOCAL_MODEL', 'llama3.1:8b')
CLOUD_MODEL = os.environ.get('CLOUD_MODEL', 'z-ai/glm-5.3-flash')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
REQUEST_TIMEOUT = 30


# ─────────────────────────────────────────────
# ROUTE DECISION
# ─────────────────────────────────────────────
class RouteDecision:
    """The result of routing: which backend, why, and which tier decided."""

    def __init__(self, backend: str, reason: str, tier: str, model: str):
        self.backend = backend      # "local" or "cloud"
        self.reason = reason        # human-readable explanation
        self.tier = tier            # "heuristic" or "classifier" or "fallback"
        self.model = model          # actual model id to use

    def to_dict(self) -> dict:
        return {
            'backend': self.backend,
            'reason': self.reason,
            'tier': self.tier,
            'model': self.model,
        }


# ─────────────────────────────────────────────
# TIER 1: HEURISTIC RULES
# ─────────────────────────────────────────────
# Signals that a question needs real reasoning / synthesis -> cloud.
CLOUD_KEYWORDS = (
    'compare', 'analyz', 'evaluate', 'strategy', 'synthesize', 'report',
    'essay', 'in depth', 'in-depth', 'step by step', 'pros and cons',
    'trade-off', 'tradeoff', 'explain why', 'research paper', 'recommend',
)

# Signals that a question is a simple lookup -> local.
LOCAL_KEYWORDS = (
    'weather', 'temperature', 'forecast',
    'news', 'latest',
    'wikipedia', 'who is', 'who was', 'when did', 'where is',
    'calculate', 'what is', 'capital of',
)

LONG_QUESTION_CHARS = 200  # questions longer than this lean toward complexity


def heuristic_route(question: str):
    """Instant rule-based routing. Returns 'local', 'cloud', or None (ambiguous)."""
    q = question.lower().strip()

    # Any cloud keyword wins immediately — synthesis tasks need the big model.
    if any(kw in q for kw in CLOUD_KEYWORDS):
        return 'cloud'
    # Simple, short lookup questions go local.
    if any(kw in q for kw in LOCAL_KEYWORDS) and len(q) <= LONG_QUESTION_CHARS:
        return 'local'
    # Very short questions are almost always simple.
    if len(q) <= 60:
        return 'local'
    # Very long questions are almost always complex.
    if len(q) > LONG_QUESTION_CHARS:
        return 'cloud'
    # Anything in between is ambiguous — let Tier 2 decide.
    return None


# ─────────────────────────────────────────────
# TIER 2: LOCAL MODEL AS CLASSIFIER
# ─────────────────────────────────────────────
CLASSIFIER_PROMPT = """You are a router for a research agent. Classify the complexity of the user's question.

Respond with ONLY valid JSON, no other text:
{"complexity": <1-10>, "route": "local" or "cloud"}

Rules:
- complexity 1-4: simple factual lookups, weather, news, math, one-tool questions -> "local"
- complexity 5-10: multi-step reasoning, comparisons, analysis, synthesis, writing -> "cloud"

Question: """


def classify_with_local_model(question: str):
    """Ask the local 8B model to rate question complexity.

    Returns 'local', 'cloud', or None if the local model is unavailable
    or gives an unparseable answer.
    """
    try:
        content, _meta = ollama_chat([{
            'role': 'user',
            'content': CLASSIFIER_PROMPT + question,
        }])
    except Exception:
        return None

    if not content:
        return None

    match = re.search(r'"route"\s*:\s*"(local|cloud)"', content)
    return match.group(1) if match else None


# ─────────────────────────────────────────────
# PUBLIC ROUTING API
# ─────────────────────────────────────────────
def route(question: str) -> RouteDecision:
    """Route a question to the right backend."""
    # Tier 1: heuristics first — free and instant.
    verdict = heuristic_route(question)
    if verdict == 'cloud':
        return RouteDecision('cloud', 'Heuristic match: complex/synthesis question', 'heuristic', CLOUD_MODEL)
    if verdict == 'local':
        return RouteDecision('local', 'Heuristic match: simple lookup question', 'heuristic', LOCAL_MODEL)

    # Tier 2: ambiguous — ask the local model to classify.
    verdict = classify_with_local_model(question)
    if verdict == 'cloud':
        return RouteDecision('cloud', 'Local classifier: rated complex', 'classifier', CLOUD_MODEL)
    if verdict == 'local':
        return RouteDecision('local', 'Local classifier: rated simple', 'classifier', LOCAL_MODEL)

    # Fail safe: no local model, no clear verdict -> cloud can always handle it.
    return RouteDecision('cloud', 'Fallback: router could not decide (local model unavailable or ambiguous)', 'fallback', CLOUD_MODEL)


# ─────────────────────────────────────────────
# MODEL CALL WRAPPERS
# ─────────────────────────────────────────────
def ollama_chat(messages):
    """Call the local Ollama model. Returns (content, meta). Raises on failure.

    meta carries token usage, latency, and generation throughput, which
    Ollama reports directly in its chat response.
    """
    start = time.perf_counter()
    response = requests.post(
        f'{OLLAMA_URL}/api/chat',
        json={'model': LOCAL_MODEL, 'messages': messages, 'stream': False},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    latency = time.perf_counter() - start

    tokens_out = data.get('eval_count', 0)
    eval_seconds = data.get('eval_duration', 0) / 1e9  # Ollama reports nanoseconds
    meta = {
        'model': data.get('model', LOCAL_MODEL),
        'tokens_in': data.get('prompt_eval_count', 0),
        'tokens_out': tokens_out,
        'latency_s': round(latency, 2),
        'tokens_per_sec': round(tokens_out / eval_seconds, 1) if eval_seconds else None,
    }
    return data['message']['content'].strip(), meta


def openrouter_chat(messages):
    """Call the cloud model via OpenRouter. Returns (content, meta). Raises on failure."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise RuntimeError('OPENROUTER_API_KEY not set')

    start = time.perf_counter()
    response = requests.post(
        OPENROUTER_URL,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        json={
            'model': CLOUD_MODEL,
            'messages': messages,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    latency = time.perf_counter() - start

    usage = data.get('usage', {})
    tokens_out = usage.get('completion_tokens', 0)
    meta = {
        'model': data.get('model', CLOUD_MODEL),
        'tokens_in': usage.get('prompt_tokens', 0),
        'tokens_out': tokens_out,
        'latency_s': round(latency, 2),
        # Throughput is approximate: it includes network overhead, unlike Ollama's
        # eval_duration, which measures generation time only.
        'tokens_per_sec': round(tokens_out / latency, 1) if tokens_out and latency else None,
    }
    return data['choices'][0]['message']['content'].strip(), meta


def call_model(messages, backend: str):
    """Dispatch to the right backend; fall back to cloud if local fails.

    Returns (content, meta).
    """
    if backend == 'local':
        try:
            return ollama_chat(messages)
        except Exception as e:
            print(f'   ⚠️  Local model failed ({e}); falling back to cloud.')
            return openrouter_chat(messages)
    return openrouter_chat(messages)