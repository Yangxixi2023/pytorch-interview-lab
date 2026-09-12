"""生成全量可运行参考文件与 GitHub 文档。不会收集练习数据。"""
import ast
from pathlib import Path

from catalog import PROBLEMS
from interview_questions import QUESTIONS
from tensor_guide import TENSOR_GUIDE

ROOT = Path(__file__).resolve().parent


def reference_source(problem, example=False):
    tree = ast.parse(problem['reference'])
    used = {node.id for node in ast.walk(tree) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load)}
    if example:
        used.add('torch')
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            node.names = [alias for alias in node.names if (alias.asname or alias.name.split('.')[0]) in used]
    tree.body = [node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom)) or node.names]
    source = ast.unparse(tree)+'\n'
    if example:
        header = problem['title']+'\n\n输入：\n'
        header += '\n'.join(f"{p['name']}: {p['type']} {p['shape']} — {p['description']}" for p in problem['inputs'])
        header += '\n\n返回（多项按元组顺序）：\n'
        header += '\n'.join(f"{p['name']}: {p['type']} {p['shape']} — {p['description']}" for p in problem['returns'])
        source = '"""'+header+'\n"""\n\n'+source
        fixture = ast.unparse(ast.parse(problem['cases'][0]['setup']))
        source += '\nif __name__ == "__main__":\n    torch.manual_seed(17)\n'
        source += '\n'.join('    '+line for line in fixture.splitlines())+'\n    print(solve(*args))\n'
    return source


def generate():
    docs=ROOT/'docs'; docs.mkdir(exist_ok=True)
    refs=ROOT/'references'; refs.mkdir(exist_ok=True)
    index=['# 全题库参考代码','',f'全部 {len(PROBLEMS)} 道题均提供可独立运行的 `.py` 文件。每个文件包含输入/返回说明、完整实现和一个示例。','',
           '从项目根目录运行，例如：`python references/ppo_loss.py`（请使用已安装依赖的虚拟环境 Python）。','',
           '[张量变换完整教程](tensor_operations.py)','']
    (refs/'tensor_operations.py').write_text(TENSOR_GUIDE,encoding='utf-8')
    lines=['# 全题库：输入、返回与参考实现','',f'{len(PROBLEMS)} 道题；每个场景使用三个固定种子执行。','',
           '| # | 题目 | 分类 | 难度 | 场景数 | 参考实现 |','| --- | --- | --- | --- | --- | --- |']
    for i,p in enumerate(PROBLEMS.values(),1):
        path=refs/(p['id']+'.py'); path.write_text(reference_source(p,True),encoding='utf-8')
        index.append(f"- [{p['title']}]({path.name})")
        lines.append(f"| {i} | {p['title']} | {p['category']} | {p['difficulty']} | {len(p['cases'])} | [{p['id']}.py](../references/{p['id']}.py) |")
    for p in PROBLEMS.values():
        lines+=['',f"## {p['title']}",'','```python',f"def solve({p['signature']}):",'    ...','```','',
                '| 参数 | 类型 | Shape | 含义 |','| --- | --- | --- | --- |']
        lines += [f"| `{v['name']}` | {v['type']} | `{v['shape']}` | {v['description']} |" for v in p['inputs']]
        lines+=['','返回：','']+[f"- `{v['name']}`：{v['type']}，`{v['shape']}`。{v['description']}" for v in p['returns']]
        lines+=['','多项返回时按上面顺序组成元组。' if len(p['returns'])>1 else '单项直接返回，不包装成元组。','',p['statement']]
    (refs/'README.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    (docs/'PROBLEMS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    qa=['# 训练岗位面试问答','',f'{len(QUESTIONS)} 道知识问答，按机器学习、预训练、后训练分类。根据原始论文与官方文档整理，不声称来自特定公司的真实面试。回答与追问由本项目重新组织；来源用于核对技术背景。','']
    for q in QUESTIONS:
        qa += [f"## {q['id']} · {q['category']} · {q['title']}",'',q['answer'],'','回答要点：','']+['- '+s for s in q['points']]
        qa += ['','追问：'+q['followup'],'','易错点：'+q['pitfall'],'','资料：'+' · '.join(f"[{s['title']}]({s['url']})" for s in q['sources']),'']
    (docs/'INTERVIEW_QA.md').write_text('\n'.join(qa),encoding='utf-8')
    print(f'Exported {len(PROBLEMS)} references, tensor guide and {len(QUESTIONS)} interview answers.')


if __name__=='__main__':
    generate()
