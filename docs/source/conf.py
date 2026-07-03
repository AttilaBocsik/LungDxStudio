# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))  # Vagy ahol az src van a docs-hoz képest
extensions = [
    'sphinx.ext.autodoc',  # Kinyeri a docstringeket a kódból
    'sphinx.ext.napoleon', # Ez kezeli a Google-stílusú kommenteket
    'sphinx.ext.viewcode', # Linket tesz a forráskódhoz
    'sphinx_rtd_theme',    # A modern téma
    'myst_parser',         # README.md integrálása
]

# Téma beállítása
html_theme = 'sphinx_rtd_theme'

# Ha a Sphinx nem találja a PyQt6-ot a gépén, add hozzá ezt:
autodoc_mock_imports = ["PyQt6", "requests", "pydicom"]
