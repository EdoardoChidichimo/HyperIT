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
from unittest.mock import Mock
sys.path.insert(0, os.path.abspath('..'))

sys.modules['jpype'] = Mock()
sys.modules['org.jpype.javadoc'] = Mock()
sys.modules['org.jpype.javadoc.JavadocExtractor'] = Mock()

from hyperit import HyperIT
HyperIT.setup_JVM(jarLocation=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'infodynamics.jar'))

import mock

MOCK_MODULES = ['org.jpype.javadoc.JavadocExtractor']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Project information -----------------------------------------------------

project = 'hyperit'
copyright = '2024, Edoardo Chidichimo'
author = 'Edoardo Chidichimo'

# The full version, including alpha/beta/rc tags
release = 'v1.0.0'


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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/utils.py']


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


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'English'


def skip_java_classes(app, what, name, obj, skip, options):
    try:
        java_keywords = ['java', 'JArray', 'JDouble', 'Javadoc']
        object_representation = str(obj)
        if any(keyword in object_representation for keyword in java_keywords):
            return True
    except Exception as e:
        print(f"Skipping {name} due to error: {e}")
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_java_classes)
