# docs_tutorial/conf.py

project = "DeepXube Tutorial"
author = "Forest Agostinelli"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_prev_next": True,
    "show_toc_level": 2,
    "github_url": "https://github.com/forestagostinelli/deepxube",
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "forestagostinelli",
    "github_repo": "deepxube",
    "github_version": "main",
    "doc_path": "docs_tutorial",
}

html_static_path = ["_static"]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

bibtex_bibfiles = ["references.bib"]