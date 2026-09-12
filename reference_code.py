"""保留参考实现原有排版与注释，仅移除文件说明和演示入口。"""
import ast


def extract_solution(source):
    module = ast.parse(source)
    lines = source.splitlines(keepends=True)
    start_line = 0
    end_line = len(lines)

    if module.body and isinstance(module.body[0], ast.Expr):
        first_value = module.body[0].value
        if isinstance(first_value, ast.Constant) and isinstance(first_value.value, str):
            start_line = module.body[0].end_lineno

    for statement in module.body:
        if not isinstance(statement, ast.If):
            continue
        condition = statement.test
        if isinstance(condition, ast.Compare) and isinstance(condition.left, ast.Name):
            if condition.left.id == '__name__':
                end_line = statement.lineno - 1
                break

    return ''.join(lines[start_line:end_line]).strip() + '\n'
