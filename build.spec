# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

# Növeljük a rekurziós limitet, mert a Dask elemzésekor a Python néha kifut belőle
sys.setrecursionlimit(5000)

block_cipher = None

# 1. Adatok és metaadatok gyűjtése
datas = []
binaries = []
hiddenimports = [
    'sklearn.metrics',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'dask.dataframe',
    'dask.distributed',
    'xgboost',
    'scipy.special.cython_special',
    'pydicom.encoders.gdcm',
    'pydicom.encoders.pylibjpeg',
    'pandas._libs.tslibs.timedeltas',
    'pyarrow',
    'src.core.learning.training_logic',
    'src.core.data_manager',
    'src.core.processing.tumor_processor',
    'src.core.learning.feature_extractor',
    'src.core.data_prep.annotation_parser'
]

# KRITIKUS: A metaadatok másolása. E nélkül a Dask és a PyArrow nem fogják látni egymást!
datas += copy_metadata('pandas')
datas += copy_metadata('pyarrow')
datas += copy_metadata('dask')
datas += copy_metadata('numpy')
datas += copy_metadata('scikit-learn')
datas += copy_metadata('xgboost')

# 2. Csomagok teljes begyűjtése
packages_to_collect = ['qfluentwidgets', 'xgboost', 'dask', 'pandas', 'sklearn', 'pyarrow']
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
    hiddenimports=hiddenimports,
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
    debug=False, # Ha még mindig nem indul, állítsd True-ra, hogy lásd a konzolhibát!
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # IDEIGLENESEN állítsd True-ra! Így látni fogod a hibaüzenetet, ha elszáll.
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