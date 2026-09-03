# Hybrid Research Agent

A research agent that answers questions using tools (web search, Wikipedia,
news, weather, calculator) — running on a **hybrid model architecture**: a
router decides whether each question is handled by a free local Ollama model
or escalated to a cloud model via OpenRouter.

```
question → Router
   ├─ heuristics (instant rules)        → simple lookups → local 8B
   ├─ local classifier (for ambiguous)  → rated complex   → cloud
   └─ any failure / Ollama down                          → cloud (fail-safe)
then agent loop: model emits JSON → code runs the tool → repeat → final answer
```

## Setup

```bash
pip install -r requirements.txt

# 1. Install and start Ollama, then pull the local model:
ollama pull llama3.1:8b

# 2. Set your API keys (copy .env.example and fill in what you have):
#    OPENROUTER_API_KEY, OPENWEATHER_API_KEY, NEWS_API_KEY
```

## Run

**Web UI** (recommended): a chat page with routing badges and step traces.

```bash
python web_app.py        # → http://localhost:5000
```

**CLI**: the original terminal loop.

```bash
python research_agent.py
```

## Files

| File | What it is |
|---|---|
| `research_agent.py` | The agent: tool loop, JSON protocol, five tools, CLI |
| `router.py` | Hybrid router: heuristics → local classifier → cloud fallback |
| `web_app.py` | Flask server exposing the agent over HTTP |
| `templates/index.html` | Single-file chat UI (no build step) |
| `TUTORIAL.md` | Study & tutorial guide — how it all works and why |

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | cloud model access (required for cloud routing) |
| `CLOUD_MODEL` | `z-ai/glm-5.3-flash` | any OpenRouter model id |
| `LOCAL_MODEL` | `llama3.1:8b` | any locally pulled Ollama model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OPENWEATHER_API_KEY` | — | weather tool |
| `NEWS_API_KEY` | — | news tool |

## Study guide

See [TUTORIAL.md](TUTORIAL.md) for a full walkthrough of the agent loop, the
router's two-tier design, and exercises to extend it.