# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

# Növeljük a limitet az összetett importok miatt
sys.setrecursionlimit(5000)

block_cipher = None

datas = []
binaries = []
hiddenimports = [
    'sklearn.metrics',
    'dask.dataframe',
    'dask.distributed',
    'dask_expr',          # ÚJ: A dask új motorja
    'dask_expr._expr',    # ÚJ: Specifikus almodulok
    'pyarrow',
    'xgboost',
    'pandas',
    'src.core.learning.training_logic',
    'src.core.data_manager',
    'src.core.processing.tumor_processor'
]

# Metaadatok kényszerítése - e nélkül a Dask nem látja a verziókat
packages_to_metadata = ['pandas', 'pyarrow', 'dask', 'xgboost', 'scikit-learn']
for pkg in packages_to_metadata:
    datas += copy_metadata(pkg)

# Teljes csomag begyűjtés
packages_to_collect = ['qfluentwidgets', 'xgboost', 'dask', 'dask_expr', 'pyarrow']
for package in packages_to_collect:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['src/gui/main_window.py'],
    pathex=[os.getcwd()],
    binaries=binaries,
    datas=datas,
    hiddenimports=list(set(hiddenimports)), # Duplikációk kiszűrése
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LungDxStudioPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Hagyd TRUE-n, amíg nem látjuk a GUI-t!
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/app_icon.ico' if os.path.exists('resources/app_icon.ico') else None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LungDxStudioPro',
)