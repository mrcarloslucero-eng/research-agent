"""Flask web UI for the hybrid research agent.

Run:  python web_app.py   then open http://localhost:5000
"""
import sys
from flask import Flask, render_template, request, jsonify

# Windows consoles may default to a code page that can't print the emoji/log
# output the agent produces; force UTF-8 so dev logging doesn't crash the app.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from research_agent import run_agent
from router import route as route_question

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask():
    question = (request.get_json(silent=True) or {}).get('question', '').strip()
    if not question:
        return jsonify({'error': 'Empty question.'}), 400

    try:
        result = run_agent(question)
    except Exception as e:
        return jsonify({'error': f'Agent error: {e}'}), 500

    return jsonify({
        'answer': result['answer'],
        'route': result['route'],
        'steps': result['steps'],
        'usage': result['usage'],
    })


@app.route('/api/route', methods=['POST'])
def preview_route():
    """Show which backend the router would pick, without running the agent."""
    question = (request.get_json(silent=True) or {}).get('question', '').strip()
    if not question:
        return jsonify({'error': 'Empty question.'}), 400
    return jsonify(route_question(question).to_dict())


if __name__ == '__main__':
    print("\n🌐 Research Agent UI → http://localhost:5000\n")
    app.run(host='127.0.0.1', port=5000, debug=False)