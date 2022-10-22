# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_rtd_theme

project = 'Neural language processing (NLP)'
copyright = '2022, Stefan W. Moeller'
author = 'Stefan W. Moeller'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# https://medium.com/@eikonomega/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365
# cheat sheet: https://sphinx-tutorial.readthedocs.io/cheatsheet/
# formatting of file: https://pythonhosted.org/an_example_pypi_project/sphinx.html
# formatting text: https://bashtage.github.io/sphinx-material/rst-cheatsheet/rst-cheatsheet.html



sys.path.append(os.path.abspath('../05_sphinx_test'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath("./_ext"))

extensions = ["sphinx.ext.autodoc", 'sphinx.ext.coverage', 'sphinx.ext.napoleon', "sphinx.ext.todo", "sphinx_rtd_theme"]

todo_include_todos = True


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
