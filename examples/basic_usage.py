"""
Minimal example — no framework needed.
Shows how Driftlock wraps the OpenAI client with zero boilerplate.

Run:
    OPENAI_API_KEY=sk-... python examples/basic_usage.py
"""

import json
import os

from driftlock import DriftlockClient, DriftlockConfig

client = DriftlockClient(
    api_key=os.environ["OPENAI_API_KEY"],
    config=DriftlockConfig(log_json=False),  # human-readable logs for the terminal
)

# Identical call signature to openai.OpenAI().chat.completions.create()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    _dl_endpoint="math_demo",
)

print("\nReply:", response.choices[0].message.content)

# Pull aggregated stats from local SQLite
print("\n--- Stats ---")
print(json.dumps(client.stats(), indent=2))
