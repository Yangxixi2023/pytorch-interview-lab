"""生成全量可运行参考文件与 GitHub 文档。不会收集练习数据。"""
from pathlib import Path

from catalog import PROBLEMS
from interview_questions import QUESTIONS
from tensor_guide import TENSOR_GUIDE

ROOT = Path(__file__).resolve().parent


def reference_source(problem, example=False):
    if example:
        return (ROOT/'references'/(problem['id']+'.py')).read_text(encoding='utf-8')
    return problem['reference']


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
        path=refs/(p['id']+'.py')
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
    qa=['# 训练岗位面试问答','',f'{len(QUESTIONS)} 道问答，按机器学习、预训练、后训练分类。','']
    for q in QUESTIONS:
        qa += [f"## {q['title']}",'',q['answer'],'','资料：'+' · '.join(f"[{s['title']}]({s['url']})" for s in q['sources']),'']
    (docs/'INTERVIEW_QA.md').write_text('\n'.join(qa),encoding='utf-8')
    print(f'Exported {len(PROBLEMS)} references, tensor guide and {len(QUESTIONS)} interview answers.')


if __name__=='__main__':
    generate()
