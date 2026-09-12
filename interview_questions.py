"""问题与直接回答，来源保留用于核对。"""
import json
from pathlib import Path

QUESTIONS = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
