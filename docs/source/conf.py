# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Vagy ahol az src van a docs-hoz képest
project = 'LungDxStudio'
copyright = '2026, Attila Bocsik'
author = 'Attila Bocsik'
release = 'Tüdő karcinómák megállapító gépi tanulási modell készítő.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Kód beolvasása
    'sphinx.ext.napoleon',     # Google/NumPy style támogatás (EZ FONTOS!)
    'sphinx.ext.viewcode',     # Link a forráskódhoz
]

templates_path = ['_templates']
exclude_patterns = []

language = 'hu'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
