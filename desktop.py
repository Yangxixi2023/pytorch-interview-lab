"""独立窗口入口。Python 解释器和运行库由 Windows 便携包携带。"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import traceback

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--smoke-test', type=Path)
    args = parser.parse_args()
    data_dir = args.data_dir or Path(os.environ['LOCALAPPDATA']) / 'PyTorchInterviewLab'
    data_dir.mkdir(parents=True, exist_ok=True)

    # pythonw 没有控制台，将启动错误记录到应用自己的数据目录。
    log = (data_dir / 'desktop.log').open('a', encoding='utf-8', buffering=1)
    sys.stdout = log
    sys.stderr = log

    # 在创建任何窗口前启用每显示器 DPI，避免 Windows 位图拉伸文字。
    import ctypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))

    from PySide6.QtCore import QLockFile, QTimer, QUrl, Qt
    from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence, QFont, QGuiApplication
    from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
    import server as backend
    from desktop_state import DesktopState

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv[:1])
    app.setFont(QFont('Microsoft YaHei UI', 10))
    app.setApplicationName('手撕实验室')
    lock = QLockFile(str(data_dir / 'application.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.information(None, '手撕实验室', '软件已经在运行，请切换到已打开的窗口。')
        return 0

    store = DesktopState(data_dir / 'progress.json')
    try:
        store.read()
    except (OSError, ValueError) as exc:
        QMessageBox.critical(None, '无法读取进度', f'{exc}\n\n数据文件：{store.path}')
        return 1

    http_server = backend.ThreadingHTTPServer(('127.0.0.1', 0), backend.Handler)
    http_server.state_store = store
    server_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    origin = f'http://127.0.0.1:{http_server.server_port}'

    class Page(QWebEnginePage):
        def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
            if navigation_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
                if url.scheme() in ('http', 'https') and not url.toString().startswith(origin + '/'):
                    QDesktopServices.openUrl(url)
                    return False
            return super().acceptNavigationRequest(url, navigation_type, is_main_frame)

        def createWindow(self, window_type):
            # 论文等 target=_blank 链接由系统浏览器打开。
            external = QWebEnginePage(self.profile(), self)
            def open_external(url):
                if url.scheme() in ('http', 'https'):
                    QDesktopServices.openUrl(url)
                external.deleteLater()
            external.urlChanged.connect(open_external)
            return external

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('手撕实验室')
            self.resize(1420, 920)
            self.setMinimumSize(900, 650)
            self.setWindowIcon(QIcon(str(APP_ROOT / 'app.ico')))
            self.closing = False
            self.ready_to_close = False
            self.profile = QWebEngineProfile('InterviewLab', self)
            self.profile.setPersistentStoragePath(str(data_dir / 'browser'))
            self.profile.setCachePath(str(data_dir / 'cache'))
            self.profile.downloadRequested.connect(self.download)
            self.view = QWebEngineView(self)
            self.view.setPage(Page(self.profile, self.view))
            settings = self.view.settings()
            settings.setFontFamily(QWebEngineSettings.FontFamily.StandardFont, 'Microsoft YaHei UI')
            settings.setFontFamily(QWebEngineSettings.FontFamily.SansSerifFont, 'Microsoft YaHei UI')
            settings.setFontFamily(QWebEngineSettings.FontFamily.FixedFont, 'Consolas')
            settings.setFontSize(QWebEngineSettings.FontSize.DefaultFontSize, 16)
            settings.setFontSize(QWebEngineSettings.FontSize.DefaultFixedFontSize, 16)
            settings.setFontSize(QWebEngineSettings.FontSize.MinimumFontSize, 12)
            self.setCentralWidget(self.view)
            self.view.load(QUrl(origin + '/'))

            application_menu = self.menuBar().addMenu('软件')
            exit_action = QAction('退出', self)
            exit_action.setShortcut(QKeySequence('Ctrl+Q'))
            exit_action.triggered.connect(self.close)
            application_menu.addAction(exit_action)

            data_menu = self.menuBar().addMenu('进度')
            export_action = data_menu.addAction('导出备份…')
            export_action.triggered.connect(self.export_progress)
            import_action = data_menu.addAction('导入备份…')
            import_action.triggered.connect(self.import_progress)
            folder_action = data_menu.addAction('打开数据文件夹')
            folder_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_dir))))

        def download(self, request):
            filename, _ = QFileDialog.getSaveFileName(self, '保存文件', request.downloadFileName())
            if filename:
                path = Path(filename)
                request.setDownloadDirectory(str(path.parent))
                request.setDownloadFileName(path.name)
                request.accept()
            else:
                request.cancel()

        def export_progress(self):
            filename, _ = QFileDialog.getSaveFileName(self, '导出进度', '手撕实验室进度.json', 'JSON (*.json)')
            if filename:
                def save_export(text):
                    if text is not None:
                        Path(filename).write_text(text, encoding='utf-8')
                self.view.page().runJavaScript('window.interviewLabExport()', save_export)

        def import_progress(self):
            filename, _ = QFileDialog.getOpenFileName(self, '导入进度', '', 'JSON (*.json)')
            if not filename:
                return
            try:
                state = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
                if not isinstance(state, dict) or not isinstance(state.get('drafts', {}), dict):
                    raise ValueError('这不是有效的练习进度文件')
                answer = QMessageBox.question(self, '导入进度', '用这个备份替换当前进度？')
                if answer == QMessageBox.StandardButton.Yes:
                    payload = json.dumps(state, ensure_ascii=False)
                    self.view.page().runJavaScript('window.interviewLabImport(' + payload + ')')
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, '导入失败', str(exc))

        def closeEvent(self, event):
            if self.ready_to_close:
                event.accept()
                return
            event.ignore()
            if self.closing:
                return
            self.closing = True
            self.view.page().runJavaScript(
                'window.interviewLabExport ? window.interviewLabExport() : null',
                self.finish_close,
            )

        def finish_close(self, state_text):
            try:
                final_state = json.loads(state_text) if state_text else None
                store.finish(final_state)
            except (ValueError, OSError) as exc:
                self.closing = False
                QMessageBox.warning(self, '进度保存失败', f'{exc}\n请先导出备份，再退出。')
                return
            backend.cancel_all_jobs()
            http_server.shutdown()
            http_server.server_close()
            self.ready_to_close = True
            self.close()

    window = Window()
    window.show()
    smoke = None
    if args.smoke_test:
        from desktop_smoke import DesktopSmoke
        smoke = DesktopSmoke(window, origin, data_dir, args.smoke_test)
        window.view.loadFinished.connect(smoke.loaded)
        QTimer.singleShot(60000, smoke.timeout)

    def exception_handler(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb, file=log)
        QMessageBox.critical(window, '软件错误', f'{exc}\n详细信息已写入：{data_dir / "desktop.log"}')
    sys.excepthook = exception_handler
    exit_code = app.exec()
    backend.cancel_all_jobs()
    lock.unlock()
    return exit_code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
