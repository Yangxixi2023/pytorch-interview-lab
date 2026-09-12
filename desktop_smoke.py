"""便携桌面版集成测试入口，仅在 --smoke-test 参数下运行。"""
import json
import os
from pathlib import Path
import threading
import time
import traceback
from urllib.request import Request, urlopen

from PySide6.QtCore import QObject, QTimer, Signal


class DesktopSmoke(QObject):
    finished = Signal(dict)

    def __init__(self, window, origin, data_dir, report_path):
        super().__init__(window)
        self.window = window
        self.origin = origin
        self.data_dir = data_dir
        self.report_path = report_path
        self.running = False
        self.completed = False
        self.loaded_count = 0
        self.restored = False
        self.font_info = {}
        self.finished.connect(self.finish)

    def loaded(self, ok):
        if not ok:
            self.finish({'error':'页面加载失败'})
            return
        if not self.running:
            QTimer.singleShot(250, self.inspect)

    def inspect(self):
        self.window.view.page().runJavaScript('''JSON.stringify({
          ready: typeof window.interviewLabExport === 'function',
          rows: document.querySelectorAll('#problem-list tr').length,
          font: getComputedStyle(document.body).fontFamily,
          fontSize: getComputedStyle(document.body).fontSize,
          state: window.interviewLabExport ? JSON.parse(window.interviewLabExport()) : null
        })''', self.inspected)

    def inspected(self, text):
        if self.running or self.completed:
            return
        state = json.loads(text)
        if not state['ready'] or state['rows'] != 89:
            QTimer.singleShot(250, self.inspect)
            return
        self.font_info = {'family':state['font'], 'size':state['fontSize']}
        if self.loaded_count == 0:
            self.restored = state['state'].get('desktop_smoke_marker') == 'persistent'
            self.loaded_count += 1
            state['state']['desktop_smoke_marker'] = 'persistent'
            state['state'].setdefault('drafts', {})['ppo_loss'] = '# 桌面版持久化验证\n'
            payload = json.dumps(state['state'], ensure_ascii=False)
            self.window.view.page().runJavaScript('window.interviewLabImport(' + payload + ')')
            return
        self.running = True
        threading.Thread(target=self.exercise_backend, daemon=True).start()

    def exercise_backend(self):
        import server
        from catalog import PROBLEMS

        def request(path, payload=None):
            body = json.dumps(payload).encode() if payload is not None else None
            headers = {'Content-Type':'application/json','X-Lab-Token':server.TOKEN}
            with urlopen(Request(self.origin+path,data=body,headers=headers),timeout=10) as response:
                return json.load(response)

        def run(payload):
            job = request('/api/run', payload)
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                state = request('/api/job/' + job['id'])
                if state['status'] == 'done':
                    return state['result']
                time.sleep(0.1)
            raise RuntimeError('判题没有按期结束')

        try:
            ppo = run({'problem':'ppo_loss','mode':'submit','code':PROBLEMS['ppo_loss']['reference']})
            if not ppo['passed']:
                raise RuntimeError(str(ppo))
            einops = run({'problem':'mha','mode':'debug','code':PROBLEMS['mha']['reference'],
                          'custom':"from einops import rearrange\nimport numpy as np\nx=torch.ones(2,3,4)\nprint(rearrange(x,'b t d -> b d t').shape)\nprint(np.arange(3))"})
            if not einops['passed']:
                raise RuntimeError(str(einops))

            child_pid_path = self.data_dir/'worker-child.pid'
            if child_pid_path.exists():
                child_pid_path.unlink()
            code = ('import subprocess, sys, time\nfrom pathlib import Path\n'
                    'def solve(x, dim=-1):\n'
                    '    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(300)"])\n'
                    f'    Path({str(child_pid_path)!r}).write_text(str(child.pid))\n'
                    '    time.sleep(300)\n')
            hanging = request('/api/run', {'problem':'softmax','mode':'run','code':code})
            deadline = time.monotonic()+20
            while not child_pid_path.exists() and time.monotonic()<deadline:
                time.sleep(0.1)
            if not child_pid_path.exists():
                raise RuntimeError('未能创建用于退出验证的子进程')
            with server.LOCK:
                worker_pid = server.JOBS[hanging['id']]['process'].pid
            self.finished.emit({'ok':True,'port':int(self.origin.rsplit(':',1)[1]),
                'gui_pid':os.getpid(),'worker_pid':worker_pid,'child_pid':int(child_pid_path.read_text()),
                'ppo_tests':len(ppo['cases']),'einops_output':einops['stdout'],
                'restored':self.restored,'font':self.font_info,'state_file':str(self.data_dir/'progress.json')})
        except Exception as exc:
            self.finished.emit({'error':str(exc),'traceback':traceback.format_exc()})

    def finish(self, report):
        if self.completed:
            return
        self.completed = True
        self.report_path.parent.mkdir(parents=True,exist_ok=True)
        self.window.grab().save(str(self.report_path.with_suffix('.png')))
        self.report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.window.close()

    def timeout(self):
        if not self.completed:
            self.finish({'error':'桌面测试超时'})
