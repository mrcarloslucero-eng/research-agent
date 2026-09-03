import requests
import json
import os
import re
import ast
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
MAX_STEPS = 5
REQUEST_TIMEOUT = 30  # seconds for Ollama
API_TIMEOUT = 10      # seconds for external APIs
MAX_CONTEXT_MESSAGES = 12  # Keep last N messages to protect context window


try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("Install duckduckgo-search: pip install duckduckgo-search")

try:
    import wikipedia
except ImportError:
    raise ImportError("Install wikipedia: pip install wikipedia")

from router import route as route_question, call_model


# ─────────────────────────────────────────────
# SAFE MATH EVALUATION (replaces dangerous eval)
# ─────────────────────────────────────────────
class SafeMathEvaluator:
    """AST-based safe evaluator for basic arithmetic."""

    _operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if b != 0 else float('inf'),
        ast.Pow: lambda a, b: a ** b,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    @classmethod
    def evaluate(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return cls.evaluate(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants allowed")
        elif isinstance(node, ast.BinOp):
            left = cls.evaluate(node.left)
            right = cls.evaluate(node.right)
            op_type = type(node.op)
            if op_type not in cls._operators:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            return cls._operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = cls.evaluate(node.operand)
            op_type = type(node.op)
            if op_type not in cls._operators:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return cls._operators[op_type](operand)
        elif isinstance(node, ast.Call):
            raise ValueError("Function calls are not allowed")
        elif isinstance(node, ast.Name):
            raise ValueError("Variables are not allowed")
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str) -> str:
    """Safely evaluate a mathematical expression using AST."""
    # Whitelist check
    allowed_chars = set('0123456789+-*/(). ')
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Only basic math characters allowed: 0-9 + - * / ( ) and spaces")

    try:
        tree = ast.parse(expression, mode='eval')
        result = SafeMathEvaluator.evaluate(tree)
        return str(result)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


# ─────────────────────────────────────────────
# OLLAMA CONNECTION
# ─────────────────────────────────────────────
def ask_model(messages: List[Dict[str, str]], backend: str = 'local') -> str:
    """Send messages to the routed model backend and return the response."""
    try:
        return call_model(messages, backend)
    except Exception as e:
        print(f"   ❌ Model request failed: {e}")
        return '{"action": "answer", "response": "Error: Could not reach any model backend (local or cloud)."}'


# ─────────────────────────────────────────────
# JSON EXTRACTION
# ─────────────────────────────────────────────
def extract_json(text: str) -> Optional[str]:
    """Extract a JSON object from model output, handling markdown and plain text."""
    if not text:
        return None

    # Try markdown code block first (```json ... ```)
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if md_match:
        return md_match.group(1)

    # Try to find the first balanced JSON object
    # This is more robust than simple find('{') for nested structures
    start = text.find('{')
    if start == -1:
        return None

    # Track brace depth to find the matching close brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    return None


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
def search_web(query: str) -> str:
    """Search DuckDuckGo for current information."""
    print(f"\n   🔍 Searching web for: '{query}'")
    if not query or not query.strip():
        return "Error: Empty search query."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"Web search error: {e}"

    if not results:
        return "No results found."

    formatted = []
    for r in results:
        title = r.get('title', 'No title')
        body = r.get('body', 'No snippet')
        formatted.append(f"Title: {title}\nSnippet: {body}")

    return "\n\n".join(formatted)


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual and historical topics."""
    print(f"\n   📖 Searching Wikipedia for: '{query}'")
    if not query or not query.strip():
        return "Error: Empty Wikipedia query."

    try:
        result = wikipedia.summary(query, sentences=5)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            first_option = e.options[0]
            return wikipedia.summary(first_option, sentences=5)
        except Exception as inner_e:
            options = ', '.join(e.options[:5])
            return f"Ambiguous query. Wikipedia options: {options}. Error: {inner_e}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for that query."
    except Exception as e:
        return f"Wikipedia error: {e}"


def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap."""
    print(f"\n   🌤️  Getting weather for: '{city}'")
    if not city or not city.strip():
        return "Error: City name is required."

    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        return "Weather API key not found. Set OPENWEATHER_API_KEY environment variable."

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                'q': city,
                'appid': api_key,
                'units': 'imperial'
            },
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Weather request failed: {e}"
    except json.JSONDecodeError:
        return "Weather API returned invalid JSON."

    try:
        return (
            f"City: {data['name']}\n"
            f"Condition: {data['weather'][0]['description']}\n"
            f"Temperature: {data['main']['temp']}°F\n"
            f"Feels like: {data['main']['feels_like']}°F\n"
            f"Humidity: {data['main']['humidity']}%\n"
            f"Wind: {data['wind']['speed']} mph"
        )
    except (KeyError, IndexError) as e:
        return f"Weather data parsing error: {e}"


def search_news(query: str) -> str:
    """Search recent news articles using NewsAPI."""
    print(f"\n   📰 Searching news for: '{query}'")
    if not query or not query.strip():
        return "Error: Empty news query."

    api_key = os.environ.get('NEWS_API_KEY')
    if not api_key:
        return "News API key not found. Set NEWS_API_KEY environment variable."

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                'q': query,
                'apiKey': api_key,
                'sortBy': 'publishedAt',
                'pageSize': 5,
                'language': 'en'
            },
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"News request failed: {e}"
    except json.JSONDecodeError:
        return "News API returned invalid JSON."

    if response.status_code != 200:
        message = data.get('message', 'Unknown error') if isinstance(data, dict) else 'Unknown error'
        return f"News lookup failed: {message}"

    articles = data.get('articles', [])
    if not articles:
        return "No news articles found for that query."

    results = []
    for article in articles:
        title = article.get('title', 'No title')
        source = article.get('source', {}).get('name', 'Unknown source')
        desc = article.get('description', 'No description')
        published = article.get('publishedAt', 'Unknown date')
        url = article.get('url', '')
        results.append(f"Title: {title}\nSource: {source}\nPublished: {published}\nURL: {url}\nSummary: {desc}")

    return "\n\n".join(results)


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    print(f"\n   🧮 Calculating: '{expression}'")
    if not expression or not expression.strip():
        return "Error: Empty expression."

    try:
        return safe_eval(expression)
    except ValueError as e:
        return f"Math error: {e}"
    except Exception as e:
        return f"Math error: {e}"


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a research assistant with access to five tools. You must respond with ONLY valid JSON. No markdown, no explanations outside JSON.

Available tools:
1. get_weather(city: str) — gets current weather for any city
2. search_web(query: str) — searches the internet for current information
3. search_wikipedia(query: str) — searches Wikipedia for factual and historical topics
4. search_news(query: str) — searches recent news articles
5. calculate(expression: str) — performs math (e.g. "100 * 1.07" or "25 / 4")

Tool Selection Rules — follow strictly:
- Weather questions → ALWAYS use get_weather. NEVER use search_web for weather. NEVER.
- Math questions → ALWAYS use calculate, never guess
- Historical or factual topics → use search_wikipedia
- Current news or recent events → use search_news
- Anything else → use search_web

After a tool returns a result:
- If get_weather returns weather data, IMMEDIATELY respond with {"action": "answer", ...}. Do not call any more tools.
- If search_news returns articles, summarize them and IMMEDIATELY respond with {"action": "answer", ...}. Do not call any more tools.
- Trust every tool result. If it returned data, use it — do not search again.
- Only output JSON. No extra text before or after.

To use a tool, respond ONLY with valid JSON:
{"action": "get_weather", "city": "McKinney, TX"}
{"action": "search_web", "query": "your query here"}
{"action": "search_wikipedia", "query": "topic here"}
{"action": "search_news", "query": "topic here"}
{"action": "calculate", "expression": "10 * 25"}

When you have enough information, respond ONLY with:
{"action": "answer", "response": "your full answer here"}
"""


# ─────────────────────────────────────────────
# CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────
def trim_messages(messages: List[Dict[str, str]], max_messages: int = MAX_CONTEXT_MESSAGES) -> List[Dict[str, str]]:
    """Trim message history to protect context window, keeping system and latest messages."""
    if len(messages) <= max_messages:
        return messages

    # Always keep system prompt (first message) and the most recent messages
    system_msg = messages[0] if messages[0].get('role') == 'system' else None
    recent = messages[-(max_messages - (1 if system_msg else 0)):]

    if system_msg:
        return [system_msg] + recent
    return recent


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────
def run_agent(user_question: str, route_decision=None):
    """Run the agent loop. Returns a dict with the answer, routing info, and step trace."""
    # ── ROUTING: pick the model backend for this question ──
    if route_decision is None:
        route_decision = route_question(user_question)
    backend = route_decision.backend
    rd = route_decision.to_dict()
    print(f"\n{'='*50}")
    print(f"❓ Question: {user_question}")
    print(f"🔀 Router: {backend.upper()} ({rd['model']}) — {rd['reason']} [tier: {rd['tier']}]")
    print(f"{'='*50}")

    steps = []  # trace for the web UI

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_question}
    ]

    for step in range(MAX_STEPS):
        print(f"\n[Step {step + 1}] Asking model what to do next...")

        # Protect context window
        messages = trim_messages(messages)

        raw_output = ask_model(messages, backend)
        print(f"   🧠 Model said: {raw_output}")

        # Extract JSON from model output
        json_str = extract_json(raw_output)
        if not json_str:
            print("   ⚠️  No valid JSON found in model output.")
            print(f"   Raw output was: {raw_output}")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": "Error: You must respond with ONLY valid JSON. No extra text. Try again using the exact format specified."
            })
            continue

        try:
            decision = json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parse error: {e}")
            print(f"   Attempted to parse: {json_str}")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": f"Error: Invalid JSON. Parse error: {e}. Respond with valid JSON only."
            })
            continue

        action = decision.get('action')
        steps.append({'step': step + 1, 'action': action,
                      'detail': decision.get('query') or decision.get('city')
                      or decision.get('expression') or ''})

        if action == 'search_web':
            query = decision.get('query', '')
            # Redirect weather queries the model mistakenly sent to search_web
            weather_keywords = ('weather', 'temperature', 'forecast', 'humidity', 'feels like', 'wind speed')
            if any(kw in user_question.lower() for kw in weather_keywords):
                print("   ↩️  Redirecting weather query from search_web → get_weather")
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": "Error: For weather questions you MUST use get_weather, not search_web. Respond with: {\"action\": \"get_weather\", \"city\": \"<city name>\"}"
                })
                continue
            if not query:
                result = "Error: Missing 'query' parameter for search_web."
            else:
                result = search_web(query)
            print(f"   📄 Got {len(result)} characters of results")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Tool result (search_web):\n{result}"})

        elif action == 'search_wikipedia':
            query = decision.get('query')
            if not query:
                result = "Error: Missing 'query' parameter for search_wikipedia."
            else:
                result = search_wikipedia(query)
            print(f"   📄 Got {len(result)} characters from Wikipedia")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Tool result (search_wikipedia):\n{result}"})

        elif action == 'get_weather':
            city = decision.get('city')
            if not city:
                result = "Error: Missing 'city' parameter for get_weather."
            else:
                result = get_weather(city)
            print(f"   📄 Weather data: {result[:100]}...")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Tool result (get_weather):\n{result}"})

        elif action == 'search_news':
            query = decision.get('query')
            if not query:
                result = "Error: Missing 'query' parameter for search_news."
            else:
                result = search_news(query)
            print(f"   📄 Got {len(result)} characters of news results")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Tool result (search_news):\n{result}"})

        elif action == 'calculate':
            expression = decision.get('expression')
            if not expression:
                result = "Error: Missing 'expression' parameter for calculate."
            else:
                result = calculate(expression)
            print(f"   📄 Result: {result}")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Tool result (calculate):\n{result}"})

        elif action == 'answer':
            response = decision.get('response', 'No response provided.')
            print(f"\n{'='*50}")
            print(f"✅ Final Answer:\n")
            print(response)
            print(f"{'='*50}\n")
            return {'answer': response, 'route': rd, 'steps': steps}

        else:
            available = "get_weather, search_web, search_wikipedia, search_news, calculate, answer"
            error_msg = f"Unknown action: '{action}'. Available actions: {available}."
            print(f"   ⚠️  {error_msg}")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": f"Error: {error_msg}\nPlease use one of the available actions and respond with valid JSON only."
            })

    # Max steps reached without an answer
    print("\n⚠️  Agent stopped — max steps reached without a final answer.")
    print("   Attempting to generate a summary from available information...")

    # One last attempt: ask model to summarize what it knows
    messages.append({
        "role": "user",
        "content": "You have reached the step limit. Please use the 'answer' action now to provide the best answer you can based on the information gathered so far."
    })
    messages = trim_messages(messages)
    raw_output = ask_model(messages, backend)

    json_str = extract_json(raw_output)
    if json_str:
        try:
            final = json.loads(json_str, strict=False)
            if final.get('action') == 'answer':
                response = final.get('response', 'No response.')
                print(f"\n{'='*50}")
                print(f"✅ Final Answer (from step limit):\n")
                print(response)
                print(f"{'='*50}\n")
                return {'answer': response, 'route': rd, 'steps': steps}
        except Exception:
            pass

    print(f"\n{'='*50}")
    print("❌ Could not obtain a final answer within the step limit.")
    print(f"{'='*50}\n")
    return {'answer': 'Sorry — I could not produce a final answer within the step limit.', 'route': rd, 'steps': steps}


if __name__ == "__main__":
    print("\n🤖 Research Agent Ready (type 'quit' to exit)")
    print("="*50)
    while True:
        try:
            question = input("\nAsk your research agent something: ")
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down agent.")
            break

        if question.lower().strip() in ['quit', 'exit', 'q']:
            print("Shutting down agent.")
            break
        if question.strip():
            run_agent(question)