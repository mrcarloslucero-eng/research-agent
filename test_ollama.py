import requests

try:
    response = requests.post(
        'http://localhost:11434/api/chat',
        json={
            'model': 'qwen2.5:7b',
            'messages': [{'role': 'user', 'content': 'Say hello in one sentence'}],
            'stream': False
        }
    )
    data = response.json()
    print(data['message']['content'])
except Exception as e:
    print(f"Error: {e}")