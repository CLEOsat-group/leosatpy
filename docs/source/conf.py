# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../..'))

from leosatpy.utils.version import __version__

now = datetime.now()
today = f"{now.year}-{now.month:02}-{now.day:02} {now.hour:02}H{now.minute:02}"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LEOSatpy'
author = 'Christian Adam'
organization = 'CLEOSat-Group'
copyright = f"2022-{now.year}, {organization}"

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    # 'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    # 'sphinx.ext.autosummary',
    "numpydoc",
    'sphinx.ext.intersphinx'
]


templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of strings:
#
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# Custom sidebar templates must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'searchbox.html'],
                 }

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'LEOSatpydoc'


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text.
# This can be an ISBN or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
