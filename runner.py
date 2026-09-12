"""Run trusted personal practice code in a disposable Python process, not a sandbox."""
import contextlib
import io
import json
import linecache
import math
import sys
import time
import traceback

import torch
from catalog import PROBLEMS

torch.set_num_threads(1)


class Output(io.StringIO):
    def write(self, s):
        room = 30000 - self.tell()
        if room > 0:
            super().write(s[:room])
        return len(s)


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value


def leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)


def compare(actual, expected, path='output'):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f'{path}: 期望 Tensor，实际 {type(actual).__name__}')
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5, check_dtype=True,
                                   msg=lambda msg: f'{path}: {msg}')
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)), f'{path}: 期望 {type(expected).__name__}'
        assert len(actual) == len(expected), f'{path}: 长度 {len(actual)} != {len(expected)}'
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f'{path}[{i}]')
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f'{path}: 字典键不一致'
        for key in expected:
            compare(actual[key], expected[key], f'{path}[{key!r}]')
    elif isinstance(expected, float):
        assert math.isclose(float(actual), expected, rel_tol=2e-4, abs_tol=2e-5), f'{path}: {actual} != {expected}'
    else:
        assert actual == expected, f'{path}: {actual!r} != {expected!r}'


def gradients(output, args):
    params = [x for x in leaves(args) if x.requires_grad]
    # A nonconstant cotangent also detects incorrect gradients of normalized outputs
    # such as softmax, whose plain output.sum() has zero gradient.
    terms = [(x*torch.linspace(0.3,1.7,x.numel(),dtype=x.dtype,device=x.device).reshape(x.shape)).sum()
             for x in leaves(output) if x.requires_grad]
    if not params:
        return []
    if not terms:
        return [None] * len(params)
    return list(torch.autograd.grad(sum(terms), params, allow_unused=True))


def summary(value):
    return repr(value)[:6000]


def evaluate(request):
    problem = PROBLEMS[request['problem']]
    source = request['code']
    filename = problem['id']+'.py'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    result = dict(cases=[], stdout='', snapshots=[])
    output = Output()
    breakpoints = set(request.get('breakpoints', []))

    def trace(frame, event, arg):
        if event == 'line' and frame.f_code.co_filename == filename and frame.f_lineno in breakpoints and len(result['snapshots']) < 40:
            result['snapshots'].append(dict(line=frame.f_lineno, locals={k:summary(v)[:800] for k,v in frame.f_locals.items() if not k.startswith('__')}))
        return trace

    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            namespace = {'__name__':'submission'}
            exec(compile(source, filename, 'exec'), namespace)
            solve = namespace['solve']
            if request.get('mode') == 'debug':
                sys.settrace(trace)
                exec(compile(request.get('custom',''), 'debug.py', 'exec'), namespace)
                result['cases'].append(dict(name='自定义调试',passed=True))
            else:
                oracle = {}
                exec(problem['reference'], oracle)
                cases = problem['cases'][:1] if request.get('mode') == 'run' else problem['cases']
                seeds = [17] if request.get('mode') == 'run' else [17,41,103]
                for seed in seeds:
                    for case in cases:
                        item = dict(name=case['name'], seed=seed, passed=False)
                        began = time.perf_counter()
                        try:
                            torch.manual_seed(seed)
                            fixture = {'torch':torch,'math':math}
                            exec(case['setup'], fixture)
                            reference_args = clone(fixture['args'])
                            user_args = clone(fixture['args'])
                            expected = oracle['solve'](*reference_args)
                            torch.manual_seed(seed)
                            sys.settrace(trace if breakpoints else None)
                            actual = solve(*user_args)
                            sys.settrace(None)
                            item.update(actual=summary(actual),expected=summary(expected),inputs=summary(fixture['args']))
                            compare(actual, expected)
                            if problem['gradient']:
                                compare(gradients(actual,user_args),gradients(expected,reference_args),'gradient')
                            item['passed'] = True
                        except Exception:
                            item['error'] = traceback.format_exc()
                        finally:
                            sys.settrace(None)
                        item['ms'] = round((time.perf_counter()-began)*1000,2)
                        result['cases'].append(item)
        except BaseException:
            result['error'] = traceback.format_exc()
        finally:
            sys.settrace(None)
    result['stdout'] = output.getvalue()
    result['passed'] = bool(result['cases']) and all(c['passed'] for c in result['cases']) and not result.get('error')
    return result


if __name__ == '__main__':
    request = json.load(sys.stdin)
    result = evaluate(request)
    sys.__stdout__.write(json.dumps(result,ensure_ascii=False))
