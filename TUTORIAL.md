# Hybrid Research Agent — Study & Tutorial Guide

This guide walks you through **how this project works and why**, step by step.
Read it alongside the source files: `research_agent.py`, `router.py`, `web_app.py`, and `templates/index.html`.

---

## Part 1 — The Agent (research_agent.py)

### The tool loop
An "agent" here is just a **loop around an LLM**. Instead of asking the model one
question and getting one answer, we:

1. Send the user's question to the model along with a **system prompt** that lists
   the available tools.
2. The model responds with a **JSON decision**, e.g.
   `{"action": "search_wikipedia", "query": "Ada Lovelace"}`.
3. Our Python code (not the model!) executes that tool.
4. We paste the tool result back into the conversation and ask again.
5. When the model has enough information, it answers with
   `{"action": "answer", "response": "..."}` and the loop ends.

Key insight: **the model decides, the code executes.** The model never runs
anything itself — it only emits JSON that we parse and act on.

### Why JSON, and not just free text?
Small local models are chatty and unreliable. The JSON contract
(`extract_json()` + `SYSTEM_PROMPT`) turns "output" into a *protocol*:

- The system prompt demands only-JSON responses.
- `extract_json()` survives markdown fences, surrounding chatter, and nested braces.
- If the model misbehaves, we **feed the error back** as a user message and let it
  retry — this self-correction loop is what makes small models workable.

### Context window management (`trim_messages`)
Every step appends tool results, so history grows fast. An 8B model has a small
context window. `trim_messages()` keeps the system prompt plus the most recent N
messages and drops the middle. Simple, but essential.

### The tools
Five tools, each a plain Python function returning a **string** (strings are what
the model can read back):

| Tool | Backing API | Needs a key? |
|---|---|---|
| `get_weather` | OpenWeatherMap | yes (`OPENWEATHER_API_KEY`) |
| `search_web` | DuckDuckGo | no |
| `search_wikipedia` | Wikipedia | no |
| `search_news` | NewsAPI | yes (`NEWS_API_KEY`) |
| `calculate` | none — AST-safe evaluator | no |

Note `safe_eval()`: never use Python's `eval()` on model output — it's arbitrary
code execution. The AST whitelist only permits numbers and `+ - * / ( )`.

---

## Part 2 — Why hybrid routing matters (router.py)

### The trade-off
| | Local 8B (Ollama) | Cloud model (OpenRouter) |
|---|---|---|
| Cost | free | paid per token |
| Latency | fast, on your machine | network round-trip |
| Quality | fine for lookups & routing | strong reasoning, synthesis, writing |
| Privacy | data stays local | data leaves your machine |

Simple questions ("weather in McKinney?") waste money and time on a frontier
model. Complex questions ("compare solar vs nuclear with pros and cons") exceed
what an 8B model reasons about well. **A router gets the best of both.**

### The two-tier design

```
question
  │
  ▼
Tier 1: heuristic_route()          ← instant, free, deterministic
  │ rules: keywords + length
  ├─ clearly simple → local
  ├─ clearly complex → cloud
  └─ ambiguous ────────┐
                       ▼
Tier 2: classify_with_local_model()  ← one cheap call to the 8B model
  │ "rate this question's complexity, respond JSON"
  ├─ local says simple → local
  ├─ local says complex → cloud
  └─ local down / garbage → CLOUD (fail-safe)
```

**Tier 1 (heuristics)** — `heuristic_route()` checks:
- *Cloud keywords*: "compare", "analyz(e)", "pros and cons", "report"… → these
  describe synthesis tasks, route straight to cloud.
- *Local keywords*: "weather", "who was", "calculate"… + short length → local.
- *Length alone*: under 60 chars → local; over 200 chars → cloud.

Heuristics are cheap, instant, and explainable — but brittle. That's fine,
because they only need to catch the obvious cases.

**Tier 2 (self-classification)** — for everything in between, we ask the local
model itself to rate complexity 1–10 and pick a route. This costs one fast 8B
call (a few hundred ms) instead of burning cloud tokens on every ambiguous case.

**Fail-safe rule**: any failure (Ollama down, unparseable classifier output, missing
key) routes to **cloud**. Why cloud? Because the cloud model can do *everything*
the local model can, but not vice versa. When in doubt, degrade upward.

This pattern has a name: **cascading fallback** — try the cheap tier first, escalate
on uncertainty or failure.

### The dispatch layer
`call_model(messages, backend)` wraps both backends behind one interface:

- `ollama_chat()` → POST to `http://localhost:11434/api/chat`
- `openrouter_chat()` → POST to `https://openrouter.ai/api/v1/chat/completions`
  with a Bearer token (OpenRouter speaks the OpenAI chat format).

One wrapper, one interface, two providers — that's what makes the agent loop
backend-agnostic. If a local call fails *mid-conversation*, it silently retries
on cloud so the user never sees a hard error.

---

## Part 3 — The Web UI (web_app.py + templates/index.html)

- `web_app.py` is a thin Flask shell: `/` serves the page, `POST /api/ask` runs
  `run_agent()` and returns the **answer + routing decision + step trace** as JSON.
  `/api/route` previews what the router would do without running the agent —
  handy for studying the router itself.
- `templates/index.html` is a single-file chat UI (no build step, no framework).
  Each answer carries a **routing badge** (⚡ local / ☁️ cloud) and a collapsible
  list of the agent's steps, so you can *see* the architecture working.

**Observability tip:** surfacing the route + trace in the UI is how you debug a
router. If an answer looks bad, the first question is always "which model handled
it, and why was it routed there?"

---

## Part 4 — Extending it (exercise roadmap)

1. **Add a tool** — copy the `search_news` pattern: function → `SYSTEM_PROMPT`
   entry → dispatch branch in the loop. (Try: currency conversion.)
2. **Confidence-based routing** — have the classifier return a confidence score;
   route ambiguous high-confidence cases differently from low-confidence ones.
3. **Route per-step, not per-question** — the final "write the answer" step could
   use cloud even when the lookups ran local.
4. **Cache tool results** — weather calls for the same city within 10 min don't
   need to hit the API twice.
5. **Streaming** — Ollama and OpenRouter both support streaming; stream tokens to
   the browser for a snappier feel.
6. **Structured output** — Ollama supports `format: 'json'` natively; try it and
   compare reliability against the prompt-only approach.

---

## Part 5 — Exercises

1. Add a `convert_currency` tool end-to-end. What three places in
   `research_agent.py` must you touch?
2. Route "explain quantum computing to a 10-year-old". Which tier decides, and
   what route comes out? Why is this case hard for heuristics?
3. What happens if `OPENROUTER_API_KEY` is unset *and* Ollama is stopped?
   Trace the code path through `route()` and `call_model()`.
4. (Stretch) Make Tier 2 classification run in parallel with Tier 1 so the
   classifier never adds latency when heuristics already decided.
   What's the trade-off?

### Suggested solutions

1. The `SYSTEM_PROMPT` (tool list + JSON example), a new `elif action == ...`
   branch in the loop, and a tool function. Also add it to the `available`
   string in the unknown-action error.
2. Tier 1 returns `None` (no keywords match, length is middling) → Tier 2. It's
   hard for heuristics because it's short and keyword-free but genuinely a
   creative/synthesis task — exactly the ambiguous zone Tier 2 exists for.
3. `route()`'s classifier call fails → tier `fallback` → cloud. Then
   `openrouter_chat()` raises `RuntimeError` (no key) → `ask_model()` catches it
   and returns an error JSON, which the loop treats as a final answer message.
   The user gets a clear error, not a crash.
4. You'd always pay the classifier latency (~0.5s) even when Tier 1 decides in
   0ms, but you'd gain a second opinion for cases where rules are wrong.
   Usually not worth it — which is why the code cascades instead.

---

## Glossary

- **Agent** — an LLM in a loop that can request tool calls until it produces a final answer.
- **Tool** — a Python function the model can invoke via JSON decisions.
- **Router** — a component that picks which model should handle a request.
- **Cascading fallback** — escalate to a stronger/more expensive tier on failure or uncertainty.
- **Heuristic** — a hand-written rule; fast and explainable, but brittle.
- **Context window** — how much text a model can consider at once.
- **JSON protocol** — treating model output as machine-parseable decisions rather than prose.
- **Fail-safe routing** — defaulting to the most capable backend when the router is unsure.