"""仅离屏验证便携包，不显示窗口、不抢占前台。"""
import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def is_alive(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 5000) == 258
    finally:
        kernel.CloseHandle(handle)


def run_check(name):
    report_path = ROOT/'.desktop-test'/(name+'.json')
    profile = ROOT/'.desktop-test'/'offscreen-profile'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen',
               QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu', QT_QUICK_BACKEND='software')
    command = [str(ROOT/'dist/InterviewLab/手撕实验室.exe'), '--data-dir', str(profile),
               '--smoke-test', str(report_path)]
    process = subprocess.Popen(command,env=env,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        exit_code = process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait()
        raise
    if exit_code:
        raise RuntimeError(f'程序退出码：{exit_code}')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not report.get('ok'):
        raise RuntimeError(str(report))
    for key in ('gui_pid','worker_pid','child_pid'):
        if is_alive(report[key]):
            raise RuntimeError(f'退出后仍有进程：{key}')
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1',report['port'])) == 0:
            raise RuntimeError('退出后服务端口仍然开放')
    saved = json.loads(Path(report['state_file']).read_text(encoding='utf-8'))
    if saved['drafts']['ppo_loss'] != '# 桌面版持久化验证\n':
        raise RuntimeError('退出时草稿未正确保存')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    run_check('offscreen-first')
    restored = run_check('offscreen-reopen')
    if not restored['restored']:
        raise RuntimeError('重新打开后进度未恢复')
    print('Offscreen desktop lifecycle and persistence checks passed.')
