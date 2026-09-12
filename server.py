"""Loopback-only HTTP UI and asynchronous disposable Python jobs."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from catalog import PROBLEMS
from generate_docs import reference_source
from interview_questions import QUESTIONS
from tensor_guide import TENSOR_GUIDE

ROOT = Path(__file__).resolve().parent
EXAMPLES = json.loads((ROOT/'examples.json').read_text(encoding='utf-8'))
TOKEN = secrets.token_urlsafe(32)
JOBS = {}
LOCK = threading.Lock()


def run_job(job, request):
    started = time.perf_counter()
    try:
        env = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1')
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'runner.py')], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        with LOCK:
            job['process'] = process
            if job.get('cancelled'):
                process.kill()
        try:
            stdout, stderr = process.communicate(json.dumps(request), timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            job['result'] = dict(passed=False,error='运行超过 30 秒，进程已终止。',stdout='')
        else:
            if job.get('cancelled'):
                job['result'] = dict(passed=False,error='已停止运行。',stdout='')
            elif process.returncode:
                job['result'] = dict(passed=False,error=stderr[-12000:] or f'Python 进程退出：{process.returncode}',stdout=stdout[-30000:])
            else:
                job['result'] = json.loads(stdout)
    except Exception as exc:
        job['result'] = dict(passed=False,error=str(exc),stdout='')
    finally:
        with LOCK:
            job.pop('process',None)
            job['elapsed'] = round(time.perf_counter()-started,2)
            job['status'] = 'done'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def respond(self, data, status=200, content_type='application/json; charset=utf-8'):
        payload = json.dumps(data,ensure_ascii=False).encode() if isinstance(data,(dict,list)) else data
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(payload)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def valid_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

    def do_GET(self):
        if not self.valid_host():
            return self.respond({'error':'Invalid host'},403)
        path = urlparse(self.path).path
        if path=='/api/problems':
            return self.respond([dict(id=p['id'],title=p['title'],category=p['category'],difficulty=p['difficulty'],gradient=p['gradient'],test_count=len(p['cases'])*3) for p in PROBLEMS.values()])
        if path=='/api/interview':
            return self.respond(QUESTIONS)
        if path=='/api/tensor-guide':
            return self.respond({'code':TENSOR_GUIDE})
        if path.startswith('/api/problem/'):
            p = PROBLEMS.get(path.rsplit('/',1)[-1])
            if not p: return self.respond({'error':'题目不存在'},404)
            data = {k:v for k,v in p.items() if k!='reference'}
            data['cases'] = [dict(case, **example) for case,example in zip(p['cases'], EXAMPLES[p['id']])]
            return self.respond(data)
        if path.startswith('/api/solution/'):
            p=PROBLEMS.get(path.rsplit('/',1)[-1])
            return self.respond({'code':reference_source(p)} if p else {'error':'题目不存在'},200 if p else 404)
        if path.startswith('/api/job/'):
            with LOCK:
                job=JOBS.get(path.rsplit('/',1)[-1])
                data={k:v for k,v in job.items() if k!='process'} if job else {'error':'运行不存在'}
            return self.respond(data,200 if job else 404)
        if path=='/api/runtime':
            return self.respond({'python':sys.version.split()[0],'executable':sys.executable,'timeout':30})
        files={'/':('index.html','text/html; charset=utf-8'),'/bundle.js':('bundle.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
        if path not in files: return self.respond({'error':'Not found'},404)
        filename, mime=files[path]
        content=(ROOT/'web'/filename).read_bytes()
        if filename=='index.html': content=content.replace(b'__TOKEN__',TOKEN.encode())
        return self.respond(content,content_type=mime)

    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=200000:
                return self.respond({'error':'请求大小需在 1–200000 字节之间'},400)
            raw=self.rfile.read(length)
        except ValueError:
            return self.respond({'error':'Content-Length 不正确'},400)
        if not self.valid_host() or self.headers.get('X-Lab-Token')!=TOKEN:
            return self.respond({'error':'请刷新页面后重试'},403)
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}:
            return self.respond({'error':'Invalid origin'},403)
        path=urlparse(self.path).path
        if path.startswith('/api/cancel/'):
            with LOCK:
                job=JOBS.get(path.rsplit('/',1)[-1])
                if job and job['status']!='done':
                    job['cancelled']=True
                    if job.get('process'): job['process'].kill()
            return self.respond({'ok':True})
        if path!='/api/run': return self.respond({'error':'Not found'},404)
        try:
            request=json.loads(raw)
            if request['problem'] not in PROBLEMS: raise ValueError('题目不存在')
            if request.get('mode') not in ('run','submit','debug'): raise ValueError('运行模式不正确')
            if not isinstance(request['code'],str): raise ValueError('代码必须是文本')
        except (ValueError,KeyError) as exc:
            return self.respond({'error':str(exc)},400)
        with LOCK:
            if sum(j['status']=='running' for j in JOBS.values())>=2:
                return self.respond({'error':'已有两个任务运行中，请先停止或等待完成'},429)
            for key in list(JOBS):
                if len(JOBS)>30 and JOBS[key]['status']=='done': del JOBS[key]
            key=secrets.token_urlsafe(12)
            job={'id':key,'status':'running'}; JOBS[key]=job
        threading.Thread(target=run_job,args=(job,request),daemon=True).start()
        return self.respond({'id':key},202)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Interview Lab: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()
