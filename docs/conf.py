# Copyright (c) ONNX Project Contributors
# Adapted for onnax from onnx/ir-py docs/conf.py.
#
# SPDX-License-Identifier: Apache-2.0
"""Configuration file for the Sphinx documentation builder.

To run the documentation: python -m sphinx docs dist/html
"""

from importlib.metadata import version as distribution_version

# -- Project information -----------------------------------------------------

project = "onnax"
copyright = "2026, onnax contributors"
author = "onnax contributors"
version = distribution_version("onnax")
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_exec_code",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = []
source_suffix = [".rst", ".md"]

master_doc = "index"
language = "en"
exclude_patterns = ["_build"]
nb_execution_mode = "off"
autosummary_generate = True
pygments_style = "default"
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_theme = "furo"
html_theme_path = ["_static"]
# -- Options for graphviz ----------------------------------------------------

graphviz_output_format = "svg"
