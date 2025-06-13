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

# Mock imports for external dependencies to avoid import errors during doc generation
autodoc_mock_imports = [
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "pandas",
    "numpy",
    "sklearn",
    "scikit-learn",
    "nltk",
    "fastapi",
    "uvicorn",
    "assemblyai",    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "sentence_transformers",
    "textblob",
    "pytubefix",
    "whisper",
    "ffmpeg",
    "azure",
    "azureml",
    "mlflow",
    "tqdm",
    "termcolor",
    "tabulate",
    "protobuf",
    "sentencepiece"
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Suppress warnings for certain patterns
suppress_warnings = [
    'autosummary.import_cycle',
    'autodoc.import_object',
]

# Configure autodoc to handle errors gracefully
autodoc_inherit_docstrings = True
autodoc_typehints = 'description'

# Add any custom CSS files
html_css_files = []

# Configure HTML theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# Configure intersphinx mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# Configure autosummary
autosummary_generate_overwrite = False
autosummary_filename_map = {}

# Theme
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"  # Other popular themes: 'alabaster', 'furo'.
html_static_path = ["_static"]
