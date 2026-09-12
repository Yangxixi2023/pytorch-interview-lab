"""生成包含解释器、运行库和窗口启动器的 Windows x64 便携目录。"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(*command):
    subprocess.run([str(part) for part in command], check=True)


def build(skip_runtime):
    destination = ROOT/'dist'/'InterviewLab'
    runtime = destination/'runtime'
    application = destination/'app'
    runtime.mkdir(parents=True, exist_ok=True)
    application.mkdir(parents=True, exist_ok=True)

    if not skip_runtime:
        # 此版本为本便携包实际验证的解释器版本。
        version = '3.14.7'
        archive = ROOT/'dist'/f'python-{version}-embed-amd64.zip'
        urllib.request.urlretrieve(
            f'https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip', archive)
        with zipfile.ZipFile(archive) as package:
            package.extractall(runtime)
        (runtime/'python314._pth').write_text(
            'python314.zip\n.\nLib/site-packages\n../app\nimport site\n',encoding='utf-8')
        run(sys.executable,'-m','pip','install','--target',runtime/'Lib/site-packages',
            '--only-binary=:all:','--python-version','3.14','--implementation','cp',
            '--abi','cp314','--abi','abi3','--platform','win_amd64','pip')
        run(runtime/'python.exe','-m','pip','install','--no-compile','torch',
            '--index-url','https://download.pytorch.org/whl/cpu')
        run(runtime/'python.exe','-m','pip','install','--no-compile','numpy','einops','PySide6')

    for file in ROOT.iterdir():
        if file.is_file() and file.suffix in {'.py','.json','.txt','.md'}:
            shutil.copy2(file,application/file.name)
    for name in ('web','references','docs'):
        shutil.copytree(ROOT/name,application/name,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'packaging/app.ico',application/'app.ico')
    compiler = Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    run(compiler,'/nologo','/target:winexe','/platform:x64',
        '/reference:System.Windows.Forms.dll',
        '/win32icon:'+str(ROOT/'packaging/app.ico'),
        '/out:'+str(destination/'手撕实验室.exe'),ROOT/'packaging/Launcher.cs')
    shutil.copy2(ROOT/'packaging/使用说明.txt',destination/'使用说明.txt')
    shutil.copy2(ROOT/'packaging/DESKTOP_NOTICES.md',destination/'DESKTOP_NOTICES.md')
    shutil.copytree(ROOT/'packaging/licenses',destination/'licenses',dirs_exist_ok=True)
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-runtime',action='store_true',help='复用已下载的便携运行环境，仅更新程序')
    build(parser.parse_args().skip_runtime)
