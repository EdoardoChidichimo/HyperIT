# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
# from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../HyperIT/'))

# def setup_JVM():
#     from jpype import startJVM, getDefaultJVMPath, isJVMStarted, shutdownJVM
#     if not isJVMStarted():
#         startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=/path/to/your/infodynamics.jar")

# # Ensure the JVM is started before any autodoc processes begin
# setup_JVM()


# class Mock(MagicMock):
#     @classmethod
#     def __getattr__(cls, name):
#         return MagicMock()

# MOCK_MODULES = ['jpype', 'jpype.startJVM', 'jpype.getDefaultJVMPath', 'get_jvm_path']
# sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# def skip_jpype_members(app, what, name, obj, skip, options):
#     # Example condition to skip JPype-related members
#     # Adjust the condition based on your specific needs
#     if name.startswith('J') or type(obj).__module__ == 'jpype':
#         return True  # Skip this member
#     return None  # Let Sphinx decide for other members


# -- Project information -----------------------------------------------------

project = 'HyperIT'
copyright = '2024, Edoardo Chidichimo'
author = 'Edoardo Chidichimo'

# The full version, including alpha/beta/rc tags
release = 'development'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 4,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    app.connect('autodoc-skip-member', skip_jpype_members)
