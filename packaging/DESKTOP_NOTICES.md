# Desktop distribution notices

The portable package includes dynamically loaded third-party components. Their original license files remain in the runtime distribution and each Python package's `.dist-info` directory. The application source is available in this repository, and the runtime packages can be replaced independently.

- **CPython 3.14.7**: Python Software Foundation license. Original distribution and source: https://www.python.org/downloads/release/python-3147/ . The runtime includes `LICENSE.txt`.
- **PySide6 / Qt 6.11.2**: distributed under the applicable LGPLv3/GPL and third-party licenses supplied with these packages. Qt source and notices: https://code.qt.io/ and https://doc.qt.io/qt-6/licenses-used-in-qt.html . Python bindings source: https://code.qt.io/cgit/pyside/pyside-setup.git/ . Qt WebEngine includes Chromium and its third-party notices.
- **PyTorch 2.14 CPU**: BSD-style license and additional third-party notices, shipped with the package. Source: https://github.com/pytorch/pytorch .
- **NumPy**: BSD-style license and bundled third-party notices. Source: https://github.com/numpy/numpy .
- **einops**: MIT license. Source: https://github.com/arogozhnikov/einops .
- Other Python dependencies retain their original license files in `runtime/Lib/site-packages`.
- Browser editor and formula rendering notices are included in `app/THIRD_PARTY_NOTICES.md`.

The launcher does not encrypt or merge the runtime libraries. They remain separate DLLs and Python packages under `runtime`, so users can replace or modify those libraries and use the application with the modified versions subject to compatibility.
