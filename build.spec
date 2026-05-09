# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# 1. Összegyűjtjük a kritikus csomagok erőforrásait és metaadatait
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
    'pyarrow', # Explicit hozzáadás
    'src.core.learning.training_logic',
    'src.core.data_manager',
    'src.core.processing.tumor_processor',
    'src.core.learning.feature_extractor',
    'src.core.data_prep.annotation_parser'
]

# A csomagok listája, amiknek a metaadataira (verzióinfó) a Dask-nak szüksége van
packages_to_collect = [
    'qfluentwidgets',
    'xgboost',
    'dask',
    'pandas',
    'sklearn',
    'pyarrow' # ÚJ: Ezt is be kell gyűjteni!
]

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
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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