# build.spec
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
import sys
import os

block_cipher = None

# Összegyűjtjük a qfluentwidgets és xgboost rejtett fájljait/függőségeit
datas = []
binaries = []
hiddenimports = [
    'sklearn.metrics',
    'dask.dataframe',
    'dask.distributed',
    'xgboost',
    'scipy.special.cython_special',
    'pydicom.encoders.gdcm',
    'pydicom.encoders.pylibjpeg',
]

# QFluentWidgets és egyéb csomagok erőforrásainak begyűjtése
packages_to_collect = ['qfluentwidgets', 'xgboost', 'dask']
for package in packages_to_collect:
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['src/gui/main_window.py'],  # A belépési pont
    pathex=[os.getcwd()],        # Fontos: a gyökérkönyvtár legyen a path-ban
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
    console=False, # True, ha látni akarod a hibaüzeneteket konzolon, False ha csak GUI
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