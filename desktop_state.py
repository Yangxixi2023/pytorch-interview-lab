"""桌面版进度写入固定数据目录，不依赖浏览器端口。"""
import json
import os
import threading
from pathlib import Path


class DesktopState:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.closing = False

    def read(self):
        with self.lock:
            if not self.path.exists():
                return {}
            return json.loads(self.path.read_text(encoding='utf-8'))

    def _write(self, state):
        if not isinstance(state, dict):
            raise ValueError('进度必须是 JSON 对象')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, self.path)

    def save(self, state):
        with self.lock:
            if self.closing:
                return False
            self._write(state)
            return True

    def finish(self, final_state=None):
        # 禁止较早发出的异步保存覆盖关闭窗口时读取的最终草稿。
        with self.lock:
            if final_state is not None:
                self._write(final_state)
            self.closing = True
