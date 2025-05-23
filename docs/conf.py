# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add Your Source Code to Python Path
import os
import sys

sys.path.insert(
    0, os.path.abspath("../src")
)  # Adjust '../src' if your code lives elsewhere.

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "emotion-clf-pipeline"
copyright = "2025, NLP6"
author = "NLP6"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Enable extensions
extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other projects' documentation (e.g., Python)
    "sphinx.ext.duration",  # Measure duration of Sphinx processing
    "sphinx.ext.doctest",  # Test code snippets in the documentation
    "sphinx.ext.autosummary",  # Generate summary tables for modules/classes
    "sphinx_rtd_theme",  # The Read the Docs theme
]

# Theme
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"  # Other popular themes: 'alabaster', 'furo'.
html_static_path = ["_static"]
