"""从真实 fixture 和参考实现生成题面示例，避免手填预期值与判题漂移。"""
import inspect
import json
from pathlib import Path

import torch
from catalog import PROBLEMS

torch.set_num_threads(1)
torch.set_printoptions(precision=4, threshold=60, edgeitems=2, linewidth=72)


def generate():
    examples = {}
    for slug, problem in PROBLEMS.items():
        namespace = {}
        exec(problem['reference'], namespace)
        solve = namespace['solve']
        cases = []
        for case in problem['cases']:
            torch.manual_seed(17)
            fixture = {'torch': torch}
            exec(case['setup'], fixture)
            arguments = inspect.signature(solve).bind(*fixture['args'])
            output = solve(*fixture['args'])
            cases.append({'arguments': {name: repr(value) for name,value in arguments.arguments.items()},
                          'expected': repr(output)})
        examples[slug] = cases
    return examples


if __name__ == '__main__':
    path = Path(__file__).with_name('examples.json')
    path.write_text(json.dumps(generate(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Generated {len(PROBLEMS)} problem example sets.')
