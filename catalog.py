"""从题目元数据和可读参考文件加载题库。"""
import json
from pathlib import Path

from reference_code import extract_solution

ROOT = Path(__file__).resolve().parent
PROBLEMS = {}
for problem in json.loads((ROOT / "problem_bank.json").read_text(encoding="utf-8")):
    reference_path = ROOT / "references" / (problem["id"] + ".py")
    problem["reference"] = extract_solution(reference_path.read_text(encoding="utf-8"))
    PROBLEMS[problem["id"]] = problem
